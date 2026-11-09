// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Causal completion feedback; exact counters and a bounded joint latency sample.

use super::*;

const SAMPLE_LIMIT: usize = 64;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CompletionFeedbackSnapshot {
    pub schema_version: u8,
    pub sla: SlaThresholds,
    pub terminal_requests: u64,
    pub completed_requests: u64,
    pub good_requests: u64,
    pub good_requests_strict_itl: u64,
    pub gpu_seconds: f64,
    /// Reservoir of (TTFT ms, optional request-mean ITL ms) from this interval.
    /// None means one output token, not zero measured ITL.
    pub latency_samples: Vec<(f64, Option<f64>)>,
    pub latency_population: u64,
}

#[derive(Debug, Default)]
pub(super) struct CompletionFeedback {
    terminal: u64,
    completed: u64,
    good: u64,
    good_strict_itl: u64,
    population: u64,
    samples: Vec<(f64, Option<f64>)>,
}

impl CompletionFeedback {
    pub(super) fn observe(&mut self, stats: &TraceRequestStats, sla: SlaThresholds) {
        self.terminal += 1;
        if stats.terminal_status != Some(ReplayTerminalStatus::Completed)
            || stats.first_admit_ms.is_none()
        {
            return;
        }
        self.completed += 1;
        let output = stats.actual_output_length();
        let (Some(first), Some(last)) = (stats.first_token_ms(), stats.last_token_ms()) else {
            if sla.is_good_without_tokens(
                (stats.terminal_time_ms.unwrap() - stats.arrival_time_ms).max(0.0),
            ) {
                self.good += 1;
                self.good_strict_itl += 1;
            }
            return;
        };
        let ttft = (first - stats.arrival_time_ms).max(0.0);
        let e2e = (last - stats.arrival_time_ms).max(0.0);
        if sla.is_set() && sla.is_good(ttft, e2e, output) {
            self.good += 1;
            if output > 1 || sla.itl_ms.is_none() {
                self.good_strict_itl += 1;
            }
        }
        let itl = (output > 1).then(|| (e2e - ttft) / (output - 1) as f64);
        self.population += 1;
        // Algorithm R, using a deterministic SplitMix64 draw. No trace lookahead
        // or mutable global RNG; replaying the same prefix reproduces the sample.
        let mut draw = self.terminal.wrapping_add(0x9e3779b97f4a7c15);
        draw = (draw ^ (draw >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        draw = (draw ^ (draw >> 27)).wrapping_mul(0x94d049bb133111eb);
        draw ^= draw >> 31;
        let index = if self.samples.len() < SAMPLE_LIMIT {
            self.samples.push((ttft, itl));
            return;
        } else {
            (draw % self.population) as usize
        };
        if index < SAMPLE_LIMIT {
            self.samples[index] = (ttft, itl);
        }
    }

    pub(super) fn snapshot(
        &mut self,
        sla: SlaThresholds,
        gpu_seconds: f64,
    ) -> CompletionFeedbackSnapshot {
        CompletionFeedbackSnapshot {
            schema_version: 1,
            sla,
            terminal_requests: self.terminal,
            completed_requests: self.completed,
            good_requests: self.good,
            good_requests_strict_itl: self.good_strict_itl,
            gpu_seconds,
            latency_samples: std::mem::take(&mut self.samples),
            latency_population: std::mem::take(&mut self.population),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector() -> TraceCollector {
        let mut c = TraceCollector::default();
        c.enable_completion_feedback();
        c.set_sla_thresholds(SlaThresholds {
            ttft_ms: Some(100.0),
            itl_ms: Some(20.0),
            e2e_ms: None,
        });
        c.set_gpus_per_worker(4, 8);
        c
    }

    fn complete(c: &mut TraceCollector, id: u128, times: &[f64]) {
        let id = Uuid::from_u128(id);
        c.on_arrival(id, 0.0, 100, times.len());
        c.on_admit(id, 0.0, 0);
        for &at in times {
            c.on_token(id, at);
        }
        c.on_terminal(
            id,
            times.last().copied().unwrap_or(0.0),
            ReplayTerminalStatus::Completed,
        );
    }

    #[test]
    fn exact_feedback_matches_report_and_preserves_short_output_policy() {
        let mut c = collector();
        complete(&mut c, 1, &[100.0, 120.0]); // Both inclusive boundaries.
        complete(&mut c, 2, &[50.0, 100.0]); // ITL failure despite low TTFT.
        complete(&mut c, 3, &[101.0, 110.0]); // TTFT failure despite low ITL.
        complete(&mut c, 4, &[50.0]); // No inter-token measurement.
        complete(&mut c, 5, &[]); // Missing token latencies are not SLO successes.
        let failed = Uuid::from_u128(6);
        c.on_arrival(failed, 0.0, 100, 2);
        c.on_terminal(failed, 150.0, ReplayTerminalStatus::Canceled);
        c.on_terminal(failed, 160.0, ReplayTerminalStatus::Canceled);
        c.add_worker_seconds(3.0, 5.0);
        let sample = c.completion_feedback_snapshot().unwrap();
        assert_eq!(sample.terminal_requests, 6);
        assert_eq!(sample.completed_requests, 5);
        assert_eq!(sample.good_requests, 2);
        assert_eq!(sample.good_requests_strict_itl, 1);
        assert_eq!(sample.gpu_seconds, 52.0);
        assert_eq!(sample.latency_samples[3], (50.0, None));
        assert_eq!(sample.latency_population, 4);
        let report = c.finish();
        assert_eq!(
            sample.good_requests,
            report.goodput.unwrap().completed_requests as u64
        );
        assert!((sample.gpu_seconds - report.throughput.gpu_hours * 3600.0).abs() < 1e-9);
    }

    #[test]
    fn interval_samples_reset_but_counters_and_gpu_time_accumulate() {
        let mut c = collector();
        complete(&mut c, 1, &[50.0, 60.0]);
        c.add_worker_seconds(1.0, 2.0);
        let first = c.completion_feedback_snapshot().unwrap();
        // The runtime passes provisioned worker seconds, including drains.
        c.add_worker_seconds(2.0, 3.0);
        let next = c.completion_feedback_snapshot().unwrap();
        assert_eq!(next.terminal_requests, first.terminal_requests);
        assert_eq!(next.good_requests, first.good_requests);
        assert_eq!(next.gpu_seconds - first.gpu_seconds, 32.0);
        assert!(next.latency_samples.is_empty());
        assert_eq!(next.latency_population, 0);
    }

    #[test]
    fn latency_reservoir_is_bounded_reproducible_and_uses_the_full_interval() {
        let run = || {
            let mut c = collector();
            for i in 1..=10_000 {
                complete(&mut c, i, &[i as f64, i as f64 + 1.0]);
            }
            c.completion_feedback_snapshot().unwrap()
        };
        let a = run();
        assert_eq!(a, run());
        assert_eq!(a.latency_population, 10_000);
        assert_eq!(a.latency_samples.len(), SAMPLE_LIMIT);
        assert!(a.latency_samples.iter().any(|(ttft, _)| *ttft > 9_000.0));
    }

    #[test]
    fn feedback_survives_completed_record_retirement_in_compact_replay() {
        let mut c = collector();
        c.begin_batch_reporting();
        for i in 1..=1000 {
            complete(&mut c, i, &[50.0, 60.0]);
            c.retire_completed().unwrap();
            assert!(c.requests.is_empty());
        }
        let sample = c.completion_feedback_snapshot().unwrap();
        assert_eq!(sample.completed_requests, 1000);
        assert_eq!(sample.good_requests, 1000);
        assert_eq!(sample.latency_samples.len(), SAMPLE_LIMIT);
        c.prepare_batch_report().unwrap();
        assert_eq!(c.finish().goodput.unwrap().completed_requests, 1000);
    }
}
