//! Focused promotion-gate checks, including a real release-mode tier-0 run.

use mw_sim::promotion::{self, PromotionConfig};

#[test]
fn tier0_omni_runs_against_utility_on_a_bounded_heldout_world() {
    let model = "../../training/artifacts/ladder/tier-0/model.onnx";
    let config = PromotionConfig {
        seeds: &[101],
        agents: 2,
        ticks: 4,
    };
    let report = promotion::run_promotion(config, model).expect("tier-0 OmniSoul should load");
    assert_eq!(report.seeds, vec![101]);
    assert_eq!(report.per_seed.len(), 1);
    assert_eq!(report.aggregate.deaths.delta.n, 1);
    assert!(report.per_seed[0].utility_total_actions > 0);
    assert!(report.per_seed[0].candidate_total_actions > 0);
}

#[test]
fn promotion_default_config_is_bounded_and_preregistered() {
    let config = PromotionConfig::default();
    config.validate().expect("fixed config validates");
    assert_eq!(config.seeds, &promotion::PROMOTION_SEEDS);
    assert_eq!(config.agents, promotion::PROMOTION_AGENTS);
    assert_eq!(config.ticks, promotion::PROMOTION_TICKS);
    assert!(!config.seeds.contains(&7));
}
