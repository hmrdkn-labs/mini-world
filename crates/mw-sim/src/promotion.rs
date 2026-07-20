//! Pre-registered multi-seed promotion gate for OmniSoul.
//!
//! This module is intentionally separate from the exploratory one-seed soak:
//! the seed list, horizon, confidence-interval method, and decision thresholds
//! are part of the evaluator's contract and must not be tuned to a pilot run.

use serde::Serialize;
use std::fmt;
use std::path::Path;

use mw_neural::{Error as NeuralError, ExpertiseLevel, OmniRuntime};
use mw_agents::soul::TOOL_SLOTS;
use mw_village::{Action, TOOL_COUNT};

use crate::soak::{self, SoakConfig, SoakReport};
/// Fixed held-out world seeds.  Seed 7 (the pilot seed) is deliberately absent.
/// These worlds are paired between UtilitySoul and OmniSoul, but no world is
/// reused across a training/validation partition by this evaluator.
pub const PROMOTION_SEEDS: [u64; 8] = [101, 211, 307, 401, 503, 601, 701, 809];
/// Bounded default evaluation horizon; changing it requires a new preregistration.
pub const PROMOTION_TICKS: u64 = 4;
/// Bounded default population.
pub const PROMOTION_AGENTS: i32 = 2;
/// Approximate two-sided 95% normal critical value used by [`confidence_interval`].
pub const CI95_Z: f64 = 1.96;

/// Maximum allowed candidate degradation in death/survival rate (five percentage
/// points). This is a non-inferiority margin, not a target tuned to seed 7.
pub const DEATH_NON_INFERIORITY_MARGIN: f64 = 0.05;
/// A need delta is measured in the simulator's 0..MAX_NEED units.
pub const NEED_NON_INFERIORITY_MARGIN: f64 = 2.0;
/// Minimum meaningful improvement for one need/social dimension.
pub const NEED_MEANINGFUL_IMPROVEMENT: f64 = 2.0;
/// Minimum meaningful improvement of the balanced hunger/energy/social mean.
pub const BALANCED_MEANINGFUL_IMPROVEMENT: f64 = 0.5;

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub n: usize,
}

/// Deterministic across-seed mean and approximate two-sided 95% CI.
///
/// The sample standard deviation is used (denominator n-1); for one value the
/// interval collapses to the observed value. Empty input is rejected so a
/// missing seed can never silently pass a gate.
pub fn confidence_interval(values: &[f64]) -> ConfidenceInterval {
    assert!(!values.is_empty(), "confidence intervals need at least one seed");
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        values
            .iter()
            .map(|value| {
                let d = *value - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    let half_width = CI95_Z * variance.sqrt() / (n as f64).sqrt();
    ConfidenceInterval {
        mean,
        lower: mean - half_width,
        upper: mean + half_width,
        n,
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct MetricComparison {
    pub utility: ConfidenceInterval,
    pub candidate: ConfidenceInterval,
    /// Candidate minus UtilitySoul. Positive is better for needs/survival;
    /// negative is better for deaths.
    pub delta: ConfidenceInterval,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SeedMetrics {
    pub seed: u64,
    pub utility_deaths: usize,
    pub candidate_deaths: usize,
    pub utility_survival_rate: f64,
    pub candidate_survival_rate: f64,
    pub utility_mean_needs: [f64; 3],
    pub candidate_mean_needs: [f64; 3],
    pub utility_total_actions: u64,
    pub candidate_total_actions: u64,
    pub utility_resource_actions: u64,
    pub candidate_resource_actions: u64,
    pub utility_histogram: [u64; TOOL_SLOTS],
    pub candidate_histogram: [u64; TOOL_SLOTS],
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PromotionAggregate {
    pub deaths: MetricComparison,
    pub survival_rate: MetricComparison,
    pub hunger: MetricComparison,
    pub energy: MetricComparison,
    pub social: MetricComparison,
    pub total_actions: MetricComparison,
    pub resource_actions: MetricComparison,
    pub action_histogram: [MetricComparison; TOOL_COUNT as usize],
    pub balanced_needs: ConfidenceInterval,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct PromotionThresholds {
    pub death_non_inferiority_margin: f64,
    pub survival_non_inferiority_margin: f64,
    pub need_non_inferiority_margin: f64,
    pub meaningful_need_improvement: f64,
    pub balanced_meaningful_improvement: f64,
}

impl Default for PromotionThresholds {
    fn default() -> Self {
        Self {
            death_non_inferiority_margin: DEATH_NON_INFERIORITY_MARGIN,
            survival_non_inferiority_margin: DEATH_NON_INFERIORITY_MARGIN,
            need_non_inferiority_margin: NEED_NON_INFERIORITY_MARGIN,
            meaningful_need_improvement: NEED_MEANINGFUL_IMPROVEMENT,
            balanced_meaningful_improvement: BALANCED_MEANINGFUL_IMPROVEMENT,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PromotionDecision {
    pub deaths_non_inferior: bool,
    pub survival_non_inferior: bool,
    pub needs_social_gate: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PromotionReport {
    pub evaluator: &'static str,
    pub ci_method: &'static str,
    pub seeds: Vec<u64>,
    pub agents: i32,
    pub ticks: u64,
    pub thresholds: PromotionThresholds,
    pub per_seed: Vec<SeedMetrics>,
    pub aggregate: PromotionAggregate,
    pub decision: PromotionDecision,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertiseLevelMetrics {
    pub deaths: usize,
    pub survival_rate: f64,
    pub needs: [f64; 3],
    pub balanced: f64,
    pub resource_actions: u64,
    pub action_histogram: [u64; TOOL_SLOTS],
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertiseSeedMetrics {
    pub seed: u64,
    pub novice: ExpertiseLevelMetrics,
    pub capable: ExpertiseLevelMetrics,
    pub expert: ExpertiseLevelMetrics,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertiseLevelAggregate {
    pub deaths: ConfidenceInterval,
    pub survival_rate: ConfidenceInterval,
    pub hunger: ConfidenceInterval,
    pub energy: ConfidenceInterval,
    pub social: ConfidenceInterval,
    pub balanced: ConfidenceInterval,
    pub resource_actions: ConfidenceInterval,
    pub action_histogram: [ConfidenceInterval; TOOL_SLOTS],
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertisePromotionAggregate {
    pub novice: ExpertiseLevelAggregate,
    pub capable: ExpertiseLevelAggregate,
    pub expert: ExpertiseLevelAggregate,
    pub capable_minus_novice_balanced: ConfidenceInterval,
    pub expert_minus_capable_balanced: ConfidenceInterval,
    pub expert_minus_novice_balanced: ConfidenceInterval,
    pub expert_vs_novice_deaths: MetricComparison,
    pub expert_vs_novice_survival: MetricComparison,
    pub expert_vs_novice_hunger: MetricComparison,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertisePromotionDecision {
    pub ordered_balanced_axis: bool,
    pub adjacent_ci_lower_bounds_positive: bool,
    pub expert_vs_novice_balanced_mean_improvement: bool,
    pub expert_vs_novice_deaths_non_inferior: bool,
    pub expert_vs_novice_survival_non_inferior: bool,
    pub expert_vs_novice_hunger_non_inferior: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ExpertisePromotionReport {
    pub evaluator: &'static str,
    pub ci_method: &'static str,
    pub seeds: Vec<u64>,
    pub agents: i32,
    pub ticks: u64,
    pub thresholds: PromotionThresholds,
    pub per_seed: Vec<ExpertiseSeedMetrics>,
    pub aggregate: ExpertisePromotionAggregate,
    pub decision: ExpertisePromotionDecision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromotionConfig {
    pub seeds: &'static [u64],
    pub agents: i32,
    pub ticks: u64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            seeds: &PROMOTION_SEEDS,
            agents: PROMOTION_AGENTS,
            ticks: PROMOTION_TICKS,
        }
    }
}

impl PromotionConfig {
    pub fn validate(self) -> Result<(), PromotionError> {
        if self.seeds.is_empty() {
            return Err(PromotionError::InvalidConfig("seed set is empty"));
        }
        if self.seeds.iter().any(|seed| *seed == 7) {
            return Err(PromotionError::InvalidConfig(
                "pilot seed 7 is not allowed in promotion evaluation",
            ));
        }
        if self.agents <= 0 || self.agents > 100 {
            return Err(PromotionError::InvalidConfig(
                "agents must be in 1..=100 for a bounded evaluation",
            ));
        }
        if self.ticks == 0 || self.ticks > 10_000 {
            return Err(PromotionError::InvalidConfig(
                "ticks must be in 1..=10_000 for a bounded evaluation",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum PromotionError {
    InvalidConfig(&'static str),
    Neural(NeuralError),
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid promotion config: {message}"),
            Self::Neural(error) => write!(f, "candidate simulation failed: {error}"),
        }
    }
}
impl std::error::Error for PromotionError {}
impl From<NeuralError> for PromotionError {
    fn from(error: NeuralError) -> Self {
        Self::Neural(error)
    }
}

fn resource_actions(report: &SoakReport) -> u64 {
    [
        Action::Eat,
        Action::Work,
        Action::Give,
        Action::Pickup,
        Action::Drop,
        Action::Use,
    ]
    .iter()
    .map(|action| report.histogram[*action as usize])
    .sum()
}

fn seed_metrics(seed: u64, utility: SoakReport, candidate: SoakReport) -> SeedMetrics {
    let utility_mean_needs = utility.mean_needs();
    let candidate_mean_needs = candidate.mean_needs();
    let agents = utility.cfg.agents.max(0) as f64;
    SeedMetrics {
        seed,
        utility_deaths: utility.deaths,
        candidate_deaths: candidate.deaths,
        utility_survival_rate: (agents - utility.deaths as f64) / agents,
        candidate_survival_rate: (agents - candidate.deaths as f64) / agents,
        utility_mean_needs,
        candidate_mean_needs,
        utility_total_actions: utility.total_actions(),
        candidate_total_actions: candidate.total_actions(),
        utility_resource_actions: resource_actions(&utility),
        candidate_resource_actions: resource_actions(&candidate),
        utility_histogram: utility.histogram,
        candidate_histogram: candidate.histogram,
    }
}

fn comparison(utility: &[f64], candidate: &[f64]) -> MetricComparison {
    assert_eq!(utility.len(), candidate.len());
    let delta: Vec<f64> = candidate
        .iter()
        .zip(utility)
        .map(|(candidate, utility)| candidate - utility)
        .collect();
    MetricComparison {
        utility: confidence_interval(utility),
        candidate: confidence_interval(candidate),
        delta: confidence_interval(&delta),
    }
}

fn aggregate(per_seed: &[SeedMetrics]) -> PromotionAggregate {
    assert!(!per_seed.is_empty());
    // Non-inferiority is preregistered in percentage points, so compare
    // per-seed death rates rather than raw counts.
    let utility_deaths: Vec<f64> = per_seed.iter().map(|m| 1.0 - m.utility_survival_rate).collect();
    let candidate_deaths: Vec<f64> = per_seed.iter().map(|m| 1.0 - m.candidate_survival_rate).collect();
    let utility_survival: Vec<f64> = per_seed.iter().map(|m| m.utility_survival_rate).collect();
    let candidate_survival: Vec<f64> = per_seed.iter().map(|m| m.candidate_survival_rate).collect();
    let utility_total: Vec<f64> = per_seed.iter().map(|m| m.utility_total_actions as f64).collect();
    let candidate_total: Vec<f64> = per_seed.iter().map(|m| m.candidate_total_actions as f64).collect();
    let utility_resource: Vec<f64> = per_seed.iter().map(|m| m.utility_resource_actions as f64).collect();
    let candidate_resource: Vec<f64> = per_seed.iter().map(|m| m.candidate_resource_actions as f64).collect();
    let needs = |m: &SeedMetrics, candidate: bool, index: usize| {
        if candidate { m.candidate_mean_needs[index] } else { m.utility_mean_needs[index] }
    };
    let mut action_histogram = [MetricComparison {
        utility: confidence_interval(&[0.0]),
        candidate: confidence_interval(&[0.0]),
        delta: confidence_interval(&[0.0]),
    }; TOOL_COUNT as usize];
    for action in 0..TOOL_COUNT as usize {
        let utility: Vec<f64> = per_seed.iter().map(|m| m.utility_histogram[action] as f64).collect();
        let candidate: Vec<f64> = per_seed.iter().map(|m| m.candidate_histogram[action] as f64).collect();
        action_histogram[action] = comparison(&utility, &candidate);
    }
    let hunger_utility: Vec<f64> = per_seed.iter().map(|m| needs(m, false, 0)).collect();
    let hunger_candidate: Vec<f64> = per_seed.iter().map(|m| needs(m, true, 0)).collect();
    let energy_utility: Vec<f64> = per_seed.iter().map(|m| needs(m, false, 1)).collect();
    let energy_candidate: Vec<f64> = per_seed.iter().map(|m| needs(m, true, 1)).collect();
    let social_utility: Vec<f64> = per_seed.iter().map(|m| needs(m, false, 2)).collect();
    let social_candidate: Vec<f64> = per_seed.iter().map(|m| needs(m, true, 2)).collect();
    let balanced_delta: Vec<f64> = per_seed
        .iter()
        .map(|m| {
            (m.candidate_mean_needs.iter().sum::<f64>()
                - m.utility_mean_needs.iter().sum::<f64>())
                / 3.0
        })
        .collect();
    PromotionAggregate {
        deaths: comparison(&utility_deaths, &candidate_deaths),
        survival_rate: comparison(&utility_survival, &candidate_survival),
        hunger: comparison(&hunger_utility, &hunger_candidate),
        energy: comparison(&energy_utility, &energy_candidate),
        social: comparison(&social_utility, &social_candidate),
        total_actions: comparison(&utility_total, &candidate_total),
        resource_actions: comparison(&utility_resource, &candidate_resource),
        action_histogram,
        balanced_needs: confidence_interval(&balanced_delta),
    }
}

/// Apply the preregistered decision rule to an aggregate. The comparisons use
fn level_aggregate(rows: &[&ExpertiseLevelMetrics]) -> ExpertiseLevelAggregate {
    let ci = |values: Vec<f64>| confidence_interval(&values);
    let deaths = rows.iter().map(|row| row.deaths as f64).collect();
    let survival = rows.iter().map(|row| row.survival_rate).collect();
    let hunger = rows.iter().map(|row| row.needs[0]).collect();
    let energy = rows.iter().map(|row| row.needs[1]).collect();
    let social = rows.iter().map(|row| row.needs[2]).collect();
    let balanced = rows.iter().map(|row| row.balanced).collect();
    let resources = rows.iter().map(|row| row.resource_actions as f64).collect();
    let action_histogram = std::array::from_fn(|action| {
        ci(rows.iter().map(|row| row.action_histogram[action] as f64).collect())
    });
    ExpertiseLevelAggregate {
        deaths: ci(deaths),
        survival_rate: ci(survival),
        hunger: ci(hunger),
        energy: ci(energy),
        social: ci(social),
        balanced: ci(balanced),
        resource_actions: ci(resources),
        action_histogram,
    }
}

fn expertise_aggregate(per_seed: &[ExpertiseSeedMetrics]) -> ExpertisePromotionAggregate {
    let novice: Vec<&ExpertiseLevelMetrics> = per_seed.iter().map(|row| &row.novice).collect();
    let capable: Vec<&ExpertiseLevelMetrics> = per_seed.iter().map(|row| &row.capable).collect();
    let expert: Vec<&ExpertiseLevelMetrics> = per_seed.iter().map(|row| &row.expert).collect();
    let capable_balanced: Vec<f64> = capable.iter().map(|row| row.balanced).collect();
    let novice_balanced: Vec<f64> = novice.iter().map(|row| row.balanced).collect();
    let expert_balanced: Vec<f64> = expert.iter().map(|row| row.balanced).collect();
    // Keep death non-inferiority in the same rate units as its percentage
    // point threshold; counts remain available in each level's report.
    let deaths_novice: Vec<f64> = novice.iter().map(|row| 1.0 - row.survival_rate).collect();
    let deaths_expert: Vec<f64> = expert.iter().map(|row| 1.0 - row.survival_rate).collect();
    let survival_novice: Vec<f64> = novice.iter().map(|row| row.survival_rate).collect();
    let survival_expert: Vec<f64> = expert.iter().map(|row| row.survival_rate).collect();
    let hunger_novice: Vec<f64> = novice.iter().map(|row| row.needs[0]).collect();
    let hunger_expert: Vec<f64> = expert.iter().map(|row| row.needs[0]).collect();
    ExpertisePromotionAggregate {
        novice: level_aggregate(&novice),
        capable: level_aggregate(&capable),
        expert: level_aggregate(&expert),
        capable_minus_novice_balanced: comparison(&novice_balanced, &capable_balanced).delta,
        expert_minus_capable_balanced: comparison(&capable_balanced, &expert_balanced).delta,
        expert_minus_novice_balanced: comparison(&novice_balanced, &expert_balanced).delta,
        expert_vs_novice_deaths: comparison(&deaths_novice, &deaths_expert),
        expert_vs_novice_survival: comparison(&survival_novice, &survival_expert),
        expert_vs_novice_hunger: comparison(&hunger_novice, &hunger_expert),
    }
}

pub fn decide_expertise(
    aggregate: &ExpertisePromotionAggregate,
    thresholds: PromotionThresholds,
) -> ExpertisePromotionDecision {
    let adjacent_mean_positive = aggregate.capable_minus_novice_balanced.mean > 0.0
        && aggregate.expert_minus_capable_balanced.mean > 0.0;
    let adjacent_ci_positive = aggregate.capable_minus_novice_balanced.lower > 0.0
        && aggregate.expert_minus_capable_balanced.lower > 0.0;
    let balanced_improvement =
        aggregate.expert_minus_novice_balanced.mean >= thresholds.balanced_meaningful_improvement;
    let deaths_non_inferior =
        aggregate.expert_vs_novice_deaths.delta.upper <= thresholds.death_non_inferiority_margin;
    let survival_non_inferior = aggregate.expert_vs_novice_survival.delta.lower
        >= -thresholds.survival_non_inferiority_margin;
    let hunger_non_inferior =
        aggregate.expert_vs_novice_hunger.delta.lower >= -thresholds.need_non_inferiority_margin;
    ExpertisePromotionDecision {
        ordered_balanced_axis: adjacent_mean_positive,
        adjacent_ci_lower_bounds_positive: adjacent_ci_positive,
        expert_vs_novice_balanced_mean_improvement: balanced_improvement,
        expert_vs_novice_deaths_non_inferior: deaths_non_inferior,
        expert_vs_novice_survival_non_inferior: survival_non_inferior,
        expert_vs_novice_hunger_non_inferior: hunger_non_inferior,
        passed: adjacent_mean_positive
            && adjacent_ci_positive
            && balanced_improvement
            && deaths_non_inferior
            && survival_non_inferior
            && hunger_non_inferior,
    }
}

/// lower confidence bounds, so a boundary value is intentionally accepted.
pub fn decide(aggregate: &PromotionAggregate, thresholds: PromotionThresholds) -> PromotionDecision {
    let deaths_non_inferior = aggregate.deaths.delta.upper <= thresholds.death_non_inferiority_margin;
    let survival_non_inferior = aggregate.survival_rate.delta.lower >= -thresholds.survival_non_inferiority_margin;
    let dimensions = [
        aggregate.hunger.delta,
        aggregate.energy.delta,
        aggregate.social.delta,
    ];
    let no_material_regression = dimensions
        .iter()
        .all(|ci| ci.lower >= -thresholds.need_non_inferiority_margin);
    let one_meaningful_dimension = dimensions
        .iter()
        .any(|ci| ci.lower >= thresholds.meaningful_need_improvement);
    let needs_social_gate = (no_material_regression && one_meaningful_dimension)
        || (no_material_regression
            && aggregate.balanced_needs.lower >= thresholds.balanced_meaningful_improvement);
    PromotionDecision {
        deaths_non_inferior,
        survival_non_inferior,
        needs_social_gate,
        passed: deaths_non_inferior && survival_non_inferior && needs_social_gate,
    }
}

/// Run paired UtilitySoul/OmniSoul simulations over the fixed held-out worlds.
pub fn run_promotion(
    config: PromotionConfig,
    model_path: impl AsRef<Path>,
) -> Result<PromotionReport, PromotionError> {
    config.validate()?;
    let model_path = model_path.as_ref();
    let mut runtime = None;
    let mut per_seed = Vec::with_capacity(config.seeds.len());
    for &seed in config.seeds {
        let sim_config = SoakConfig {
            seed,
            agents: config.agents,
            ticks: config.ticks,
        };
        let utility = soak::run_with_habits(sim_config, false);
        let (candidate, next_runtime) =
            soak::run_omni_cached(sim_config, model_path, runtime)?;
        runtime = Some(next_runtime);
        per_seed.push(seed_metrics(seed, utility, candidate));
    }
    let aggregate = aggregate(&per_seed);
    let thresholds = PromotionThresholds::default();
    let decision = decide(&aggregate, thresholds);
    Ok(PromotionReport {
        evaluator: "mw-v1-omni-promotion-v1",
        ci_method: "paired across-seed mean; sample SD; normal 95% CI (z=1.96)",
        seeds: config.seeds.to_vec(),
        agents: config.agents,
        ticks: config.ticks,
        thresholds,
        per_seed,
        aggregate,
        decision,
    })
}

fn expertise_level_metrics(report: &SoakReport) -> ExpertiseLevelMetrics {
    let needs = report.mean_needs();
    ExpertiseLevelMetrics {
        deaths: report.deaths,
        survival_rate: (report.cfg.agents - report.deaths as i32) as f64 / report.cfg.agents as f64,
        needs,
        balanced: needs.iter().sum::<f64>() / 3.0,
        resource_actions: resource_actions(report),
        action_histogram: report.histogram,
    }
}

pub fn run_expertise_promotion(
    config: PromotionConfig,
    model_path: impl AsRef<Path>,
) -> Result<ExpertisePromotionReport, PromotionError> {
    config.validate()?;
    let levels = [ExpertiseLevel::Novice, ExpertiseLevel::Capable, ExpertiseLevel::Expert];
    let mut runtimes: [Option<OmniRuntime>; 3] = [None, None, None];
    let mut per_seed = Vec::with_capacity(config.seeds.len());
    for &seed in config.seeds {
        let sim_config = SoakConfig { seed, agents: config.agents, ticks: config.ticks };
        let mut rows: [Option<ExpertiseLevelMetrics>; 3] = [None, None, None];
        for (index, level) in levels.into_iter().enumerate() {
            let (report, runtime) = soak::run_omni_cached_with_expertise(
                sim_config,
                &model_path,
                level,
                runtimes[index].take(),
            )?;
            rows[index] = Some(expertise_level_metrics(&report));
            runtimes[index] = Some(runtime);
        }
        per_seed.push(ExpertiseSeedMetrics {
            seed,
            novice: rows[0].take().expect("novice result"),
            capable: rows[1].take().expect("capable result"),
            expert: rows[2].take().expect("expert result"),
        });
    }
    let aggregate = expertise_aggregate(&per_seed);
    let thresholds = PromotionThresholds::default();
    let decision = decide_expertise(&aggregate, thresholds);
    Ok(ExpertisePromotionReport {
        evaluator: "mw-v1-expertise-promotion-v1",
        ci_method: "paired across-seed mean; sample SD; normal 95% CI (z=1.96)",
        seeds: config.seeds.to_vec(),
        agents: config.agents,
        ticks: config.ticks,
        thresholds,
        per_seed,
        aggregate,
        decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(delta: f64) -> MetricComparison {
        let ci = confidence_interval(&[delta]);
        MetricComparison {
            utility: confidence_interval(&[0.0]),
            candidate: ci,
            delta: ci,
        }
    }

    fn aggregate_with_needs(hunger: f64, energy: f64, social: f64, balanced: f64) -> PromotionAggregate {
        let zero = metric(0.0);
        PromotionAggregate {
            deaths: zero,
            survival_rate: zero,
            hunger: metric(hunger),
            energy: metric(energy),
            social: metric(social),
            total_actions: zero,
            resource_actions: zero,
            action_histogram: [zero; TOOL_COUNT as usize],
            balanced_needs: confidence_interval(&[balanced]),
        }
    }

    #[test]
    fn fixed_seed_set_is_disjoint_from_pilot_and_unique() {
        assert!(!PROMOTION_SEEDS.contains(&7));
        for (i, seed) in PROMOTION_SEEDS.iter().enumerate() {
            assert!(!PROMOTION_SEEDS[..i].contains(seed));
        }
    }

    #[test]
    fn ci_is_deterministic_and_uses_sample_sd() {
        let a = confidence_interval(&[1.0, 2.0, 3.0]);
        let b = confidence_interval(&[1.0, 2.0, 3.0]);
        assert_eq!(a, b);
        assert!((a.lower - (2.0 - 1.96 / 3.0_f64.sqrt())).abs() < 1e-12);
        assert!((a.upper - (2.0 + 1.96 / 3.0_f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn threshold_boundaries_are_inclusive() {
        let mut a = aggregate_with_needs(NEED_MEANINGFUL_IMPROVEMENT, 0.0, 0.0, 0.0);
        a.deaths = metric(DEATH_NON_INFERIORITY_MARGIN);
        a.survival_rate = metric(-DEATH_NON_INFERIORITY_MARGIN);
        assert!(decide(&a, PromotionThresholds::default()).passed);
        a.hunger = metric(NEED_MEANINGFUL_IMPROVEMENT - f64::EPSILON);
        assert!(!decide(&a, PromotionThresholds::default()).needs_social_gate);
    }

    #[test]
    fn death_ci_uses_rate_units_not_raw_counts() {
        let row = SeedMetrics {
            seed: 101,
            utility_deaths: 0,
            candidate_deaths: 1,
            utility_survival_rate: 1.0,
            candidate_survival_rate: 0.99,
            utility_mean_needs: [0.0; 3],
            candidate_mean_needs: [0.0; 3],
            utility_total_actions: 0,
            candidate_total_actions: 0,
            utility_resource_actions: 0,
            candidate_resource_actions: 0,
            utility_histogram: [0; TOOL_SLOTS],
            candidate_histogram: [0; TOOL_SLOTS],
        };
        let aggregate = aggregate(&[row]);
        assert!((aggregate.deaths.delta.mean - 0.01).abs() < 1e-12);
    }

    #[test]
    fn balanced_gate_rejects_material_dimension_regression() {
        let mut a = aggregate_with_needs(5.0, -3.0, 5.0, 5.0);
        a.deaths = metric(0.0);
        a.survival_rate = metric(0.0);
        assert!(!decide(&a, PromotionThresholds::default()).needs_social_gate);
    }
}
