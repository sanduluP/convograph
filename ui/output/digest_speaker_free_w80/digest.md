# Meeting digest — `finance_speaker_free`

80 of 5964 episodes · generated 2026-09-08T13:24:26

## What this meeting was about

- **compliance** — in 62 windows · with SAR Reporting Requirements, owner, Risk: Inconsistent KYC Records, exception log
- **operations** — in 39 windows · with owner, Risk: Inconsistent KYC Records, define monitoring requirements spec, SAR Reporting Requirements
- **data** — in 38 windows · with SAR Reporting Requirements, define monitoring requirements spec, owner, source-data defects
- **engineering** — in 32 windows · with final matrix text, SAR Reporting Requirements, retrieval path, approval date
- **owner** — in 31 windows · with Ops, Finance, compliance, legal
- **data-engineering** — in 30 windows · with mapping diff, published hash, owner, timestamp
- **timestamp** — in 27 windows · with Ops, legal, data-engineering, engineering
- **legal** — in 26 windows · with Risk: Inconsistent Sustainability Standards, owner, compliance, SAR Reporting Requirements

## The documents it kept coming back to

- http://sharepoint/aml_(anti-money_laundering)_project/risk:_incomplete_regulatory_coverage_spec — in 34 windows
- http://sharepoint/aml_(anti-money_laundering)_project/investigation_workflow_definition_spec — in 25 windows
- http://sharepoint/aml_(anti-money_laundering)_project/sar_reporting_requirements_spec — in 19 windows
- http://sharepoint/aml_(anti-money_laundering)_project/model_feature_engineering_spec — in 16 windows
- SharePoint — in 8 windows

## What changed

- **compliance → AML Ops** (21d)
  - was: Compliance is aligning on any required fixes with Product, Engineering, and Ops.
  - became: If any field stays log-only, compliance will lock the fallback evidence set today.
- **Support → incident triage owner** (21d)
  - was: Support is the escalation owner.
  - became: _(successor not identified)_
- **legal → mapping exceptions** (20d)
  - was: Legal is required to review open exceptions.
  - became: Legal can post the stamp on http://sharepoint/regulatory_compliance_program/develop_training_materials_spec today.
- **compliance → rule set** (19d)
  - was: Compliance has no approved screening-rule updates since the spec refresh on http://sharepoint/aml_(anti-money_laundering)_project/cdd_data_source_integration_spec.
  - became: Compliance Alignment Review can be closed as Detected once the July 19 boundary is posted.
- **legal → consent** (18d)
  - was: Legal is reviewing identity verification.
  - became: Legal can post the stamp on http://sharepoint/regulatory_compliance_program/develop_training_materials_spec today.
- **data → risk:_esg_data_provider_reliability_spec** (18d)
  - was: Data’s confirmation of live backup coverage for the July 29 window is the last blocker I see; once they provide freshness and lineage evidence in http://sharepoint/sustainable_finance_strategy/risk:_esg_data_provider_reliability_spec, we can close the fallback path cleanly.
  - became: _(successor not identified)_
- **Reporting → ownership gaps** (18d)
  - was: Reporting must come to the July 9 working session with a named owner for any coverage gaps.
  - became: The Design ESG Screening Criteria phase should use the 2025-07-27 baseline for locking disclosures.
- **data → named owner** (16d)
  - was: freeze only the eligibility yes/no and owner fields now
  - became: _(successor not identified)_
- **Analytics → launch-critical fixes** (16d)
  - was: Analytics validates launch-critical fixes.
  - became: Analytics can freeze the drop-off readout and keep July 18 clean
- **legal → decision** (15d)
  - was: Legal is required to stamp the two items (verification and data-handling) today.
  - became: _(successor not identified)_
- **compliance → exact typology gaps** (14d)
  - was: Compliance starts the must-monitor list with Data today.
  - became: Compliance created a clean audit line by freezing the cutoff now.
- **engineering → payload mapping** (14d)
  - was: Engineering monitors payload mapping to ensure stability.
  - became: _(successor not identified)_
- **fields → baseline** (14d)
  - was: Exception fields should not be locked today; instead, use real edge cases to confirm ownership and evidence model before freezing.
  - became: _(successor not identified)_
- **tighter review path → Ops** (13d)
  - was: Tighter review path for higher-risk alerts is used by Ops.
  - became: _(successor not identified)_
- **data-engineering → lineage** (13d)
  - was: Data Engineering should track any drift in lineage.
  - became: Data Engineering monitors for drift in the untouched row.
- **owner → compliance** (12d)
  - was: Ownership of exception decisions is assigned to Compliance.
  - became: _(successor not identified)_
- **Ops → matrix note** (12d)
  - was: Ops publishes the matrix in the spec today.
  - became: _(successor not identified)_
- **parked item → named reviewer** (12d)
  - was: Each parked item should be tagged with a decision owner.
  - became: _(successor not identified)_
- **legal → owner** (11d)
  - was: Legal stamps owner in http://sharepoint/customer_onboarding_optimization/risk:_regulatory_scope_ambiguity_spec today.
  - became: DataTeam to post the one-line delta with owner, exact timestamp, and impacted fields against http://sharepoint/aml_(anti-money_laundering)_project/cdd_data_source_integration_spec today—then I’ll stamp it!
- **Ops → http://sharepoint/regulatory_compliance_program/risk:_control_gaps_identified_spec** (11d)
  - was: Ops must confirm ownership and send missing monitoring artifacts against the internal compliance audit specification by EOD tomorrow.
  - became: Ops confirms any queue ownership shifts by Thursday.
- **Reporting → Risk: Inconsistent Sustainability Standards** (11d)
  - was: Sustainability Reporting should stay blocked until the July 29 boundary is confirmed in http://sharepoint/sustainable_finance_strategy/risk:_inconsistent_sustainability_standards_spec.
  - became: Reporting confirms the July 29 boundary.
- **spec → audit trail** (11d)
  - was: That gives us a defensible boundary for July 6 and keeps the audit trail clean if someone tries a “wording-only” change later.
  - became: i’ll have `@source_owner` + Data Eng stamp `owner/default/cutoff/tested-fallback` and mark any late-arrival rows only if they touch that path in http://sharepoint/financial_reporting_automation/risk:_source_system_data_quality_spec today.
- **QA → report logic** (11d)
  - was: QA needs to verify disclosure-note edit in http://sharepoint/financial_reporting_automation/design_automated_report_templates_spec by 2025-07-19.
  - became: _(successor not identified)_
- **QA → report logic** (11d)
  - was: QA needs to verify release-path denial in http://sharepoint/financial_reporting_automation/design_automated_report_templates_spec by 2025-07-19.
  - became: _(successor not identified)_
- **QA → report logic** (11d)
  - was: QA needs to verify blocked overwrite in http://sharepoint/financial_reporting_automation/design_automated_report_templates_spec by 2025-07-19.
  - became: _(successor not identified)_

## What was settled

- `CONFIRMS` Compliance confirms ownership mappings are clean.
- `OWNS` Compliance owns the fallback evidence set if effective date stays log-only.
- `LOCKS` Compliance will lock the critical-row definition.
- `OWNS` Ops owns confirmed cases.
- `CONFIRMS` Ops confirms whether any downstream scenarios still need coverage.
- `CONFIRMS` Ops confirms propagation of the stamped master
- `OWNS` Compliance owns the register, and Ops keeps the daily row updates current.
- `CONFIRMS` Legal and Compliance must confirm the final control language before pre-tagging of fallback items proceeds.
- `FREEZES` Compliance is responsible for freezing jurisdiction-specific exceptions.
- `FREEZES` Ops freezes the listed IDs once the open-gap map lands.
- `LOCKS` Can we treat this as the final control shape and close the July 27 risk path?
- `FREEZES` Ops freezes listed IDs off that version today.
- `OWNS` Ops owns the ownership map.
- `OWNS` Compliance owns the owner.
- `OWNS` Compliance owns final call on remediation routing
- `OWNS` Compliance owns the rule decision.
- `OWNS` Compliance is the owner of exception decisions in http://sharepoint/customer_onboarding_optimization/kyc_vendor_selection_spec.
- `FREEZES` Compliance freezes only unmapped jurisdiction or scenario as the must-gap set Friday, against the one-file cut.
- `OWNS` Compliance owns the exception log owner in the risk:_esg_data_provider_reliability_spec document.
- `FREEZES` Compliance can freeze only rows with field ID, retrieval path, owner, approval date, and status right after that.

## Still open

- `REQUIRES_CONFIRMATION_FROM` Data Engineering must confirm rule owners and evidence expectations for the regulatory compliance gap spec before anything is locked down.
- `REQUIRES` compliance wrap-up requires alignment on data-match exceptions and segregation-of-duties sign-off
- `REQUIRES` Compliance requires Reporting to confirm the handoff rules in the investigation workflow definition specification before testing starts.
- `REQUIRES` Reporting requires Compliance to flag any new guidance that could shift the control design in the SAR reporting requirements specification.
- `REQUIRES` Engineering requires Compliance to pre-tag approved fallback items in the SAR reporting requirements specification.
- `REQUIRES` Engineering requires Compliance to confirm the final control language before pre-tagging fallback items.
- `NEEDS` QA needs a hard bar to prevent rerunning without control
- `REQUIRES` Compliance requires the exact gap disposition reason to be included in the matrix.
- `REQUIRES` Compliance requires that screening inputs be clean and traceable.
- `REQUIRES` Compliance requires a quick check on any final wording changes in the develop_training_materials_spec.
- `REQUIRES` Compliance requires that audit trail be maintained.
- `REQUIRES` Compliance requires that anything not meeting the signed-off criteria stays provisional.
- `REQUIRES` Compliance requires the frozen snapshot and sign-off to be tied together for closure.
- `REQUESTS` Compliance requests a leadership call on ownership for the investigation handoff.
- `REQUIRES` The hard intake gate requires blanks to be rejected.

## Who was in the room

- User_12 — 94 utterances across 63 windows (24%)
- User_9 — 86 utterances across 56 windows (22%)
- User_5 — 85 utterances across 56 windows (21%)
- User_2 — 79 utterances across 57 windows (20%)
- User_10 — 26 utterances across 23 windows (6%)
- User_13 — 16 utterances across 14 windows (4%)
- User_1 — 9 utterances across 8 windows (2%)
- User_7 — 5 utterances across 5 windows (1%)
