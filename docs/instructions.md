# Next Dev Playbook – Economics Replay & Metal Backend

The-block has just landed the Launch Governor economics gate, a binary persistence rewrite, and the entire `metal-backend/` tree. The last focused test run (`CARGO_TARGET_DIR=target_tmp cargo test --package the_block --test economics_integration`) succeeded, proving that `ChainDisk` now round-trips `economics_prev_market_metrics` and that governor streak logic respects persisted snapshots. Everything documented below assumes that baseline and expands it. Treat this file as the contract for the next development wave: every section is prescriptive, and each time you complete or re-scope work you must update this document with the same level of detail.

---

## 0. Working Agreements & Guardrails
1. **Determinism over velocity**: every economics signal (`MarketMetrics`, epoch counters, tariff snapshots, subsidy splits) must survive restart and re-computation. If you change a struct, begin with persistence + replay fixtures, then wire runtime logic.
2. **Disk is the source of truth**: no metrics, counters, or gate streaks may rely solely on in-memory state. Snapshots (`ChainDisk`) and telemetry logs must reconstruct the exact system observed prior to reboot.
3. **Telemetry before automation**: Prometheus/Grafana, CLI, and docs must expose new fields before we rely on them for governance automation. Operators should be able to verify automation with zero code spelunking.
4. **Test as you go**: add/extend integration tests in `node/tests/economics_integration.rs` for every economics persistence change, unit tests for `launch_governor`, and `cargo nextest`/`cargo test` gates in CI. For Metal Orchard changes, keep `ctest --output-on-failure` and `cargo test --package metal-ffi` (once it exists) green.
5. **Document rationale**: every major decision (codec choice, replay contract, test harness for Metal) goes in §10. Leave breadcrumbs: what changed, why, tests run.

---

## 1. What Is Already Done
- `ChainDisk` writes/reads `economics_prev_market_metrics` and all epoch-level counters via `ledger_binary::write_chain_disk`/`read_chain_disk` (`node/src/ledger_binary.rs:328-575`). All constructors (`node/src/lib.rs:2050-2941`) now populate the new fields.
- `node/tests/economics_integration.rs` covers:
  - `test_chain_disk_roundtrip_preserves_market_metrics`
  - `test_launch_governor_economics_sample_retains_metrics_after_restart`
  - `test_launch_governor_economics_gate_lifecycle`
  - Additional convergence, tariff, and shock-response scenarios.
- Launch Governor reads persisted metrics through `LiveSignalProvider::economics_sample` (`node/src/launch_governor/mod.rs:386-395`) and gates autopilot toggles accordingly.
- Telemetry gauges for epoch tx count/volume/treasury/payouts exist (`node/src/telemetry.rs:5338-5362`, `7430-7441`), but they only expose raw counters and do not yet surface previous market metrics or autopilot streaks.
- Metal Orchard builds/tests pass with vendored GoogleTest or FetchContent along with `metal_tensor_tests` covering CPU, MPS, fallback, and high-rank tensors.

---

## 2. Execution Order Overview
Work the sections in this order unless a production incident forces a swap:
1. **Finalize economics persistence surfacing** (CLI + telemetry + docs) so operators can inspect the new counters/metrics.
2. **Make market metric derivation deterministic** across storage/compute/energy/ad markets and tie replay to runtime parity.
3. **Harden Launch Governor runtime surface** (RPC, CLI, DB migrations) and write replay-vs-runtime CI jobs.
4. **Tighten ledger hygiene** (schema bump, diff tooling, fixture rotation) so new fields cannot silently regress.
5. **Modernize Metal Orchard** focusing on the future removal of GoogleTest while ensuring CI coverage.

Each subsection below lists prerequisites, concrete implementation steps, verification commands, and documentation hooks. Nothing is optional; if priorities shift, edit this file and explain why.

---

## 3. Economics Persistence Surfacing

### 3.1 Snapshot & CLI instrumentation for epoch counters
**Goal**: Operators running `tb-cli node snapshot create` must immediately see the persisted epoch counters and last-known `MarketMetrics`.

1. **Expose counters in `ChainDiskInfo` CLI output**  
   - File: `cli/src/commands/node/snapshot.rs` (or equivalent CLI module).  
   - Action: extend the JSON emitted by snapshot dumping to include `economics_epoch_tx_*`, payout buckets, and `economics_prev_market_metrics`.  
   - Format as fixed-point strings (basis points) to avoid float drift. Use the same rounding function you add for telemetry (see §3.3).

2. **Include fields in snapshot file headers**  
   - File: `node/src/lib.rs:2915+` (`Blockchain::to_chain_disk`).  
   - Action: ensure optional metadata printed during snapshot creation (logs/CLI) includes the new metrics so operators can cross-check without decoding binaries.

3. **Add acceptance test**  
   - Extend `node/tests/economics_integration.rs` with a CLI-style test: write a snapshot to a temp dir, invoke the CLI helper (use `assert_cmd` from `tests/cli` harness) to dump JSON, and compare values to the known `ChainDisk`.

4. **Docs**  
   - Update `docs/operations.md` > “Snapshots” to instruct operators to confirm non-zero `economics_prev_market_metrics` after snapshot creation. Provide sample `tb-cli` output.

5. **Verification commands**  
   - `cargo test --package the_block --test economics_integration -- --ignored snapshot_cli_roundtrip` (new test).  
   - Manual: `tb-cli node snapshot create --output /tmp/foo && tb-cli node snapshot inspect /tmp/foo`.

### 3.2 Persisted metrics → Telemetry
**Goal**: The telemetry stack should display both epoch counters and the previous market metrics to prove autopilot state survives restarts.

1. **Extend `update_economics_epoch_metrics`**  
   - File: `node/src/telemetry.rs:7430-7441`.  
   - Action: include gauges for `economics_prev_market_metrics` (four utilization + four margin series). Apply deterministic rounding (e.g., convert to millionths and log integers).  
   - Add `EconomicsPrevMetric` struct to `crate::telemetry::metrics` to reuse in RPC/gRPC payloads.

2. **Prometheus exporter**  
   - File: `monitoring/metrics/collector.rs` (or wherever metrics surfaces).  
   - Action: register new gauges `economics_prev_market_metrics_{storage,compute,energy,ad}_{utilization,margin}`. Document in `monitoring/README.md`.

3. **Grafana dashboard**  
   - Update `monitoring/output/index.json` (or `.tf`/`.json` dashboards) with a panel that shows the persisted metrics and autopilot enablement over time.  
   - Provide thresholds/annotations for gate entries/exits pulled from the same time series.

4. **Tests**  
   - Unit-test telemetry update function using deterministic floats (mock `MetricsWriter`).  
   - Add an integration test under `monitoring/tests` to confirm new gauges appear when hitting the metrics endpoint.

### 3.3 Operator runbook upgrade
**Deliverable**: Step-by-step instructions under `docs/operations.md` for verifying autopilot after snapshot, overriding autopilot, and interpreting telemetry anomalies.

1. Create a new subsection “Economics Autopilot Runbook”. Cover:  
   - Checking gate state: `tb-cli governor status` (after you implement it in §4).  
   - Snapshot inspection checklist (presence of persisted metrics).  
   - Grafana panels to watch and alert thresholds.  
   - Manual override procedure using `tb-cli governor intent submit`.
2. Include copy-paste command sequences and sample outputs.  
3. Link to `docs/monitoring.md` and `docs/governance.md` from this section so ops know where the data comes from.

---

## 4. Deterministic Market Metric Derivation
`node/src/economics/replay.rs::derive_market_metrics_from_chain` still short-circuits to placeholder metrics. We must derive each market’s utilization and provider margin deterministically from ledger data so replay matches live execution.

### 4.1 Storage market
1. **Input inventory**:  
   - Contracts/receipts: `storage_market/src/lib.rs`, `storage/tests/market_incentives.rs`.  
   - Ledger writes: search `economics_epoch_storage_payout_block`.
2. **Implementation**:  
   - Build an aggregation helper that marches over storage receipts for the replay window (bounded by epoch start/end heights).  
   - Compute: provisioned bytes, fulfilled demand, payouts, and provider cost proxies.  
   - Convert to utilization ratio (fulfilled / committed) and provider margin ((payout - cost)/cost).  
   - Persist intermediate rollups in the snapshot so replay does not require hitting sled. Consider a new `ChainDisk::storage_epoch_stats` struct.
3. **Tests**:  
   - Add fixture blocks in `tests/economics_replay_fixtures.rs`.  
   - Write `#[test] fn replay_storage_metrics_matches_live()` that runs both runtime and replay paths.

### 4.2 Compute market
1. **Input**: `node/src/compute_market/snark.rs`, scheduler logs, and compute receipts.  
2. **Implementation**:  
   - Track each compute job’s requested vs delivered compute-seconds.  
   - Compute utilization across the epoch (sum delivered / capacity).  
   - Derive provider margin from payouts minus energy/ad costs recorded in `economics_epoch_compute_payout_block`.  
   - If runtime already derives these metrics (check `update_economics_epoch_metrics`), mirror that logic and add logging to ensure replay matches.

### 4.3 Energy market
1. **Input**: `crates/energy-market/src/lib.rs`, oracle price feeds.  
2. **Implementation**:  
   - Replay emission events, convert to BLOCK using locked oracle prices, and compute net PnL per epoch.  
   - Persist enough data (oracle sample, energy sold/bought) to reconstruct after restart.  
   - Add tests referencing `energy-market/tests`.

### 4.4 Ad market
1. **Input**: `node/src/economics/ad_market_controller.rs`, `node/tests/ad_market_rpc.rs`.  
2. **Implementation**:  
   - Count filled campaigns vs reserved spend to compute utilization.  
   - Provider margin = payouts vs actual impression costs (pull from ad receipts).  
   - Document any heuristics (e.g., missing receipts) in `docs/economics.md`.

### 4.5 Replay contract & fixtures
1. **Extend `ReplayedEconomicsState`** with per-market diagnostics (source heights, totals).  
2. **Add CLI command** `tb-cli economics replay --from <snapshot>` that runs replay and prints parity summary (per metric diff).  
3. **CI**: Add `cargo test --package the_block --test economics_replay_parity` covering synthetic chains and real fixtures (`tests/data/economics/chain_slice.json`). Run under GitHub Actions macOS + Linux.

---

## 5. Launch Governor Runtime & DB Hardening

### 5.1 Governor RPC and CLI
1. **RPC layer**:  
   - File: `node/src/rpc/launch_governor.rs` (or add new module).  
   - Expose endpoint `GET /governor/status` returning: gate states, streak counters, `EconomicsSample`, autopilot enabled flag.  
   - Ensure JSON schema is versioned; include `schema_version` from `ChainDisk`.
2. **CLI**:  
   - Command `tb-cli governor status` that calls the RPC and prints both machine-readable JSON and a human summary (table with gate, streak, autopilot).  
   - Command `tb-cli governor intents --gate economics --limit 20` showing reason, metrics snapshot, and autopilot action.
3. **Tests**:  
   - Integration tests under `cli/tests/governor_status.rs` hitting a test node via `tb-test-harness`.

### 5.2 Governor DB migration
1. **Inventory**: `node/governor_db/**` contains per-gate state.  
2. **Migration tool**:  
   - Add `scripts/migrate_governor_db.rs` (Rust) or `just governor-migrate`.  
   - Behavior: scan DB, insert missing `economics` gate entries with defaults, and log actions.  
   - Run automatically on node startup (if schema < new version) and expose `--dry-run`.
3. **Docs**: update `docs/operations.md` with upgrade steps: backup DB, run migration, verify via CLI.

### 5.3 Intent audit trail
1. Extend `launch_governor::plan_intent` to attach:  
   - `EconomicsSample` snapshot (epoch counters, metrics, autopilot state).  
   - Gate decision metadata (streak counters, evaluation thresholds).  
2. Persist JSON alongside the existing reason string inside `governor_db`.  
3. Add `monitoring/tools/render_foundation_dashboard.py` support to visualize gate transitions annotated with metric snapshots.

---

## 6. Telemetry, Monitoring, Alerts

### 6.1 Prometheus alerts
Add alerting rules under `monitoring/prometheus/alerts.yml`:
1. `EconomicsTxCountFlatline`: `economics_epoch_tx_count == 0` for 2 consecutive epochs.  
2. `EconomicsRewardJump`: abs(Δ block_reward_per_block) > 20% epoch-over-epoch.  
3. `AutopilotFlap`: autopilot toggles more than twice in 24h.

Each alert should include runbook links (`docs/operations.md#economics-autopilot-runbook`).

### 6.2 Monitoring scripts
1. Ensure `monitoring/scripts/ingest_metrics.py` records persisted metrics daily for historical charts.  
2. Add CLI in `monitoring/tools` to diff two metric captures (similar to ledger diff tool) to aid debugging.

### 6.3 Documentation updates
Consolidate monitoring docs:
1. `docs/monitoring.md`: add section describing new gauges, dashboards, alerts.  
2. `docs/governance.md`: cross-link to monitoring instructions.

---

## 7. Ledger & Snapshot Hygiene

### 7.1 Schema version bump
1. Increment `state::schema::SCHEMA_VERSION` and `ChainDisk::schema_version`.  
2. Update all migrations in `node/src/lib.rs:2110-2253` to populate default `economics_prev_market_metrics` and future epoch stats.  
3. Add fixture snapshots in `tests/data/chain_disk/v12/*.bin` and update tests to cover upgrades from v11 → v12.

### 7.2 Binary diff tooling
1. Create `tools/ledger_diff/src/main.rs`:  
   - Accept two snapshot paths.  
   - Decode via `ledger_binary::decode_chain_disk`.  
   - Emit JSON diff with new fields (metrics, counters, schema).  
2. Integrate into CI:  
   - Add GitHub workflow step `cargo run -p ledger_diff -- fixtures/v12_a.bin fixtures/v12_b.bin`.  
   - Fail build if diff output misses expected keys (write golden JSON fixtures).

### 7.3 Fixture rotation
1. Regenerate `tests/data/chain_disk/*.bin` after schema bump.  
2. Update doc `docs/ECONOMIC_SYSTEM_CHANGELOG.md` detailing new fields and compatibility claims.  
3. Mention required node version in release notes under `releases/`.

---

## 8. Testing & CI Enhancements

### 8.1 Replay vs runtime CI job
1. Add GitHub Action `.github/workflows/economics-replay.yml`:  
   - Steps: `cargo fmt -- --check`, `cargo clippy --workspace -- -D warnings`, `cargo test --package the_block --test economics_integration`, `cargo test --package the_block --test economics_replay_parity`.  
   - Upload artifacts: replay parity logs, ledger diff outputs.  
   - Run on macOS (Metal support) and Linux (CPU fallback) to ensure portability.

### 8.2 `nextest` adoption
1. Configure `nextest.toml` to shard long-running economics+governor tests.  
2. Document commands in `docs/developer_handbook.md`.

### 8.3 Smoke tests for autopilot toggles
1. Add `tests/governor_autopilot_smoke.rs`: spin up ephemeral node, force metrics into “enter” then “exit” states, assert autopilot toggles and persists.  
2. Use deterministic metrics (mock providers) to avoid flakiness.

---

## 9. Metal Orchard Modernization (incl. GoogleTest Removal Plan)

### 9.1 Current context recap
- Tests live under `metal-backend/metal-tensor/tests/*.cpp` and rely on GoogleTest macros (`TEST`, `EXPECT_EQ`, etc.).  
- `CMakeLists.txt` conditionally venders `third_party/googletest` (31k LOC) or FetchContent’s zip.  
- CI coverage is manual; there is no GitHub workflow gating the Metal build.  
- The Metal tree duplicates documentation (see `metal-backend/README.md`) and includes `experimental/` assets (datasets, FlashAttention dylib).

### 9.2 Near-term hygiene (before removing GoogleTest)
1. **Add CI coverage**  
   - Workflow `.github/workflows/metal-orchard.yml`: configure with Ninja, build `orchard_metal`, run `ctest --output-on-failure`. Run on macOS (Apple Silicon) and Linux (CPU-only).  
   - Cache `~/.cache/bazel`/`~/Library/Developer` as needed to keep runtimes sane.
2. **Document test entry points** in `metal-backend/README.md`: commands for building, running tests, and environment vars (e.g., `ORCHARD_TENSOR_PROFILE`).

### 9.3 GoogleTest elimination track (low priority but defined)
**Objective**: remove the vendored GoogleTest subtree and replace the test harness with something lighter (Catch2, doctest, or custom macros).

1. **Inventory tests and required features**  
   - Files: `tensor_tests.cpp`, `multi_device_tests.cpp`, `fallback_tests.cpp`, `high_rank_tests.cpp`.  
   - Note which macros are used (fixtures? typed tests?). Currently only `TEST`/`EXPECT_*` macros, so migration is straightforward.

2. **Choose replacement harness**  
   - Preferred: [doctest](https://github.com/doctest/doctest) or `Catch2` because they are header-only, easy to vendor as a single file, and integrate with CMake.  
   - Decision criteria:  
     - Works on macOS + Linux without extra dependencies.  
     - Supports death tests? (not currently used).  
     - Permits custom reporters for CI integration.
   - Document the decision + rationale in §10 once picked.

3. **Introduce abstraction layer**  
   - Create `metal-backend/metal-tensor/tests/test_harness.h` with macros mapping `ORCHARD_TEST`, `ORCHARD_EXPECT_EQ`, etc., to the chosen framework.  
   - Replace `<gtest/gtest.h>` includes with this header. This isolates future framework swaps.

4. **Update CMake**  
   - Remove `third_party/googletest/` addition.  
   - Add FetchContent or vendored single-header for the new framework in `metal-backend/metal-tensor/tests/CMakeLists.txt`.  
   - Keep `FETCHCONTENT_FULLY_DISCONNECTED` logic for air-gapped builds.  
   - Ensure `add_test(NAME metal_tensor_tests ...)` still works.

5. **Delete vendored GoogleTest**  
   - Remove the `metal-backend/third_party/googletest` directory once tests compile/run without it.  
   - Update `.gitmodules` (if any) and `.gitignore`.

6. **Docs & Release notes**  
   - Update `metal-backend/README.md` and `AGENTS.md` describing the new harness and rationale.  
   - Mention the removal in the next release notes.

7. **Verification**  
   - `cmake -S metal-backend -B metal-backend/build -G Ninja`  
   - `cmake --build metal-backend/build --target metal_tensor_tests`  
   - `ctest --test-dir metal-backend/build --output-on-failure`

### 9.4 Additional Metal tasks (run in parallel with economics work only if bandwidth allows)
1. **Dataset/dylib hygiene**  
   - Replace tracked files `metal-backend/experimental/data/wikitext-2/wiki.train.tokens` and `metal-backend/experimental/kernel_lib/flashattn/libflash_attn.dylib` with download/build scripts (checksum verification).  
   - Update `.gitignore` and docs.
2. **Rust FFI bridge**  
   - Design how `node` will call Metal kernels. Proposed approach: create `crates/metal-ffi` using `cxx` to expose tensor APIs.  
   - Build a proof-of-concept binary under `examples/metal_bridge` showing Node → Metal inference/training call.  
   - Add tests ensuring CPU fallback works on Linux (CI can run them even without Metal).
3. **Profiling + instrumentation**  
   - Document `ORCHARD_TENSOR_PROFILE` usage (Metal + CPU).  
   - Provide script to parse `/tmp/orchard_tensor_profile.log` and integrate into CI logs when tests fail.

---

## 10. Decision Backlog (update as decisions land)
Maintain this section as a living log. Fill in the “Decision / Status / Notes” table whenever you make a call.

| Decision | Status | Notes |
| --- | --- | --- |
| `MarketMetrics` codec (binary vs serde JSON) | **Chosen**: Binary, fixed-point rounding | Implemented in `ledger_binary.rs`. Keep fixtures updated. |
| Float vs fixed-point for telemetry & CLI | **TODO** | Recommend 1e6 fixed-point integers for utilization/margins. Document conversion helpers. |
| Replay contract scope (consensus-grade vs diagnostic) | **TODO** | Decide whether replay blocks node startup on mismatch or merely logs warnings. Favor consensus-grade with CI gating. |
| Governor RPC schema versioning | **TODO** | Determine version negotiation strategy before shipping CLI. |
| Metal test harness replacement | **TODO** | Pick `doctest` or `Catch2`. Document migration plan + timeline. |
| Dataset/dylib distribution strategy for Metal experimental assets | **TODO** | Choose between scripted downloads or optional Git LFS. |

Keep this table accurate; it is the authoritative map for reviewers and future maintainers.

---

## 11. Checklist Before Handoff
Use this as the “definition of done” for the next dev session. Every item must be checked (and reflected in git history + this doc) before you hand off again.

1. [ ] `tb-cli governor status` exposes economics gate, streaks, metrics snapshot.  
2. [ ] Telemetry dashboards show persisted `MarketMetrics` and autopilot toggles with alerts.  
3. [ ] Replay harness proves parity between runtime and replay across fixture chains.  
4. [ ] Schema version bumped, fixtures regenerated, ledger diff tool committed and wired into CI.  
5. [ ] Metal Orchard CI job green on macOS + Linux; plan for GoogleTest removal documented with selected harness and macro shim.  
6. [ ] `docs/operations.md` runbook updated with CLI + telemetry instructions.  
7. [ ] This file updated with outcomes, commands run, and any new decisions.

---

### Reference Commands (keep these around)
```bash
# Economics + replay
CARGO_TARGET_DIR=target_tmp cargo test --package the_block --test economics_integration
CARGO_TARGET_DIR=target_tmp cargo test --package the_block --test economics_replay_parity
just governor-simulate   # add this Just target when you create the replay harness

# Snapshots
tb-cli node snapshot create --output /tmp/block.snap
tb-cli node snapshot inspect /tmp/block.snap

# Telemetry
python monitoring/tools/render_foundation_dashboard.py --input monitoring/output/index.json

# Metal Orchard
cmake -S metal-backend -B metal-backend/build -G Ninja
cmake --build metal-backend/build --target metal_tensor_tests
ctest --test-dir metal-backend/build --output-on-failure
```

Keep iterating on this list as tooling evolves.

---

End of playbook. Update aggressively.
