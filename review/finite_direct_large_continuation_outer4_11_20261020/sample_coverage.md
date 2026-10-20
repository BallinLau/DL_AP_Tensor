# Actual constructed sampling coverage

```json
{
  "b_K_dependence": "conditional empirical CDF by new capital quartile; original capital never copied",
  "b_eta_dependence_preserved": false,
  "b_z_dependence_preserved": false,
  "jitter": 0.0,
  "source_kind": "RECORDED_INITIAL_PARENT_DISTRIBUTION",
  "source_limitation": "Templates are actual saved initialization records used by the completed run, not realized post-transition b distributions. Dynamic micro snapshots were not saved; no distributions are reconstructed from aggregate summaries.",
  "splits": {
    "train": {
      "b_mean_max": 0.4182511116621497,
      "b_mean_min": 0.16796566538695043,
      "beta_parents": 4096,
      "conditional_fallback_firms": 0,
      "empirical_parents": 4096,
      "parent_bank_fingerprint": "cfd61de4c3f8fa4100b7a8d5aebd1f7b16375e23d990f0683269ef36987682b2",
      "parents": 8192,
      "sampling_schedule_fingerprint": "5d485128f7528c48622570b02c26c350b417fcf5596a20d83314e94dda70ba2b",
      "template_reuse_fraction": 0.0,
      "unique_empirical_templates": 4096,
      "unique_selected_empirical_CDF_shapes": 4096
    },
    "validation": {
      "b_mean_max": 0.41696061559021474,
      "b_mean_min": 0.17133475858207015,
      "beta_parents": 1024,
      "conditional_fallback_firms": 0,
      "empirical_parents": 1024,
      "parent_bank_fingerprint": "486679c9bf883afeff186aa88e38a5745d7b79871047a9a690b4edd5ba61c86b",
      "parents": 2048,
      "sampling_schedule_fingerprint": "f65fe12147815d0152a6806bdd45a5be3deacb5cc39a7cdb2ad6601eb774469d",
      "template_reuse_fraction": 0.0,
      "unique_empirical_templates": 1024,
      "unique_selected_empirical_CDF_shapes": 1024
    }
  },
  "template_leakage_audit": {
    "source_path_overlap": 0,
    "template_id_overlap": 0,
    "train_templates": 4096,
    "validation_templates": 1024
  },
  "version": "finite_direct_beta_empirical_50_v2"
}
```

These are CPU initial banks, not completed economic simulation/training rows.
