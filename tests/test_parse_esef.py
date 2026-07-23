"""Tests for scout.fundamentals.parse.esef. Parses real archived xBRL-JSON
(OIM) documents from filings.xbrl.org -- no network, no XBRL library.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import PeriodType
from scout.fundamentals.parse.esef import EsefJsonParser

_ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive" / "esef" / "2026" / "07" / "21"
_F25823 = _ARCHIVE / "25823" / "v1" / "25823.json"
_F25824 = _ARCHIVE / "25824" / "v1" / "25824.json"
_F25825 = _ARCHIVE / "25825" / "v1" / "25825.json"

_needs_25823 = pytest.mark.skipif(not _F25823.exists(), reason="real archived fixture not present")
_needs_25824 = pytest.mark.skipif(not _F25824.exists(), reason="real archived fixture not present")
_needs_25825 = pytest.mark.skipif(not _F25825.exists(), reason="real archived fixture not present")


def _parser() -> EsefJsonParser:
    return EsefJsonParser()


class TestRealFilings:
    @_needs_25823
    def test_parses_a_substantial_number_of_numeric_facts(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={"lei": "43473797", "country": "UA"})

        # 25823 carries ~426 raw facts; a large share are numeric (financial
        # statement line items) and the rest (auditor name, disclosure prose,
        # dates) are correctly skipped -- see the non-numeric test below.
        assert len(result.facts) > 200
        assert result.taxonomy == "ifrs-full"
        # For this Ukrainian filer the authoritative id is the 8-digit EDRPOU
        # code carried in the fact's own entity dimension, not a 20-char LEI --
        # here it happens to equal the harvest metadata's (mislabelled) "lei".
        assert result.entity_id == "43473797"
        assert result.entity_id.isdigit()
        assert len(result.entity_id) != 20  # i.e. not LEI-shaped

    @_needs_25823
    def test_produces_both_instant_and_duration_facts_with_correct_periods(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={})

        instant_facts = [f for f in result.facts if f.period_type is PeriodType.INSTANT]
        duration_facts = [f for f in result.facts if f.period_type is PeriodType.DURATION]
        assert instant_facts
        assert duration_facts

        # f301: PropertyPlantAndEquipment, instant period "2022-01-01T00:00:00".
        ppe = next(f for f in instant_facts if f.local_name == "PropertyPlantAndEquipment")
        assert ppe.period_start is None
        assert ppe.period_end == date(2022, 1, 1)
        assert ppe.value == 1000.0

        # f512: ReceiptsFromSalesOfGoodsAndRenderingOfServices, duration
        # "2021-01-01T00:00:00/2022-01-01T00:00:00" -> the END of the span is
        # what's stored as period_end, per the OIM convention.
        receipts = next(f for f in duration_facts if f.local_name == "ReceiptsFromSalesOfGoodsAndRenderingOfServices")
        assert receipts.period_start == date(2021, 1, 1)
        assert receipts.period_end == date(2022, 1, 1)
        assert receipts.value == 906000.0

    @_needs_25823
    def test_both_core_and_national_extension_taxonomies_are_split_correctly(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={})

        ifrs_facts = [f for f in result.facts if f.taxonomy == "ifrs-full"]
        ua_facts = [f for f in result.facts if f.taxonomy == "ua_full_ifrs_core"]
        assert ifrs_facts
        assert ua_facts

        ppe = next(f for f in ifrs_facts if f.local_name == "PropertyPlantAndEquipment")
        assert ppe.concept_key == "ifrs-full:PropertyPlantAndEquipment"
        # local_name must never retain the "taxonomy:" prefix.
        assert ":" not in ppe.local_name

    @_needs_25823
    def test_currency_is_extracted_from_the_iso4217_unit(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={})

        monetary_facts = [f for f in result.facts if f.unit is not None]
        assert monetary_facts
        assert all(f.unit == "UAH" for f in monetary_facts)

    @_needs_25823
    def test_non_numeric_facts_are_skipped_and_counted_in_a_warning(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={})

        # The auditor entity name (f136) and other disclosure-prose facts are
        # text, not numeric -- they must not appear as RawFacts.
        assert not any(f.local_name == "NameOfTheAuditEntity" for f in result.facts)
        assert any("non-numeric" in w for w in result.warnings)

    @_needs_25823
    def test_dimensioned_facts_are_flagged(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="25823", entity_hint={})

        # f687: ifrs-full:Equity broken down by ComponentsOfEquityAxis.
        dimensioned = [f for f in result.facts if f.is_dimensioned]
        assert dimensioned
        equity_breakdown = next(f for f in dimensioned if f.local_name == "Equity")
        assert equity_breakdown.is_dimensioned is True

        # The consolidated, non-dimensioned total must also survive as its
        # own fact -- the normalizer needs both.
        undimensioned = [f for f in result.facts if not f.is_dimensioned]
        assert undimensioned

    @_needs_25823
    def test_accession_is_stamped_onto_every_fact(self):
        payload = _F25823.read_bytes()
        result = _parser().parse(payload, accession="acc-25823", entity_hint={})

        assert result.accession == "acc-25823"
        assert all(f.accession == "acc-25823" for f in result.facts)

    @_needs_25825
    def test_entity_name_is_pulled_from_name_of_reporting_entity(self):
        payload = _F25825.read_bytes()
        result = _parser().parse(payload, accession="25825", entity_hint={"lei": "44356194", "country": "UA"})

        assert result.entity_id == "44356194"
        assert result.entity_name is not None
        assert result.entity_name.strip() != ""

    @_needs_25824
    def test_25824_also_parses_cleanly(self):
        payload = _F25824.read_bytes()
        result = _parser().parse(payload, accession="25824", entity_hint={})

        assert result.facts
        assert result.entity_id  # non-empty
        assert result.taxonomy == "ifrs-full"


class TestSyntheticEdgeCases:
    def test_garbage_payload_returns_empty_filing_with_warning_not_raise(self):
        result = _parser().parse(b"not json at all", accession="bad-1", entity_hint={"lei": "LEIVALUE"})

        assert result.facts == []
        assert result.has_xbrl is False
        assert result.warnings
        assert result.entity_id == "LEIVALUE"  # falls back to the manifest hint
        assert result.taxonomy == "ifrs-full"
        assert result.accession == "bad-1"

    def test_valid_json_but_not_an_xbrl_json_document_returns_empty_with_warning(self):
        result = _parser().parse(b'{"hello": "world"}', accession="bad-2", entity_hint={})

        assert result.facts == []
        assert result.warnings

    def test_json_array_instead_of_object_returns_empty_with_warning(self):
        result = _parser().parse(b"[1, 2, 3]", accession="bad-3", entity_hint={})

        assert result.facts == []
        assert result.warnings

    def test_malformed_individual_facts_are_skipped_not_fatal(self):
        payload = b"""
        {
          "documentInfo": {"namespaces": {"ifrs-full": "http://x"}},
          "facts": {
            "f1": {"value": "100", "dimensions": {"concept": "ifrs-full:Revenue", "entity": "scheme:12345678", "period": "2021-01-01T00:00:00/2022-01-01T00:00:00", "unit": "iso4217:USD"}},
            "f2": {"value": "text, unprefixed concept", "dimensions": {"concept": "NoTaxonomyPrefix", "entity": "scheme:12345678", "period": "2022-01-01T00:00:00"}},
            "f3": {"dimensions": {"concept": "ifrs-full:Assets", "entity": "scheme:12345678", "period": "garbage-not-a-date"}},
            "f4": "not even a dict"
          }
        }
        """
        result = _parser().parse(payload, accession="mix-1", entity_hint={})

        assert len(result.facts) == 1
        assert result.facts[0].local_name == "Revenue"
        assert result.entity_id == "12345678"
        assert any("malformed" in w for w in result.warnings)

    def test_instant_and_duration_period_parsing_from_synthetic_fact(self):
        payload = b"""
        {
          "facts": {
            "inst": {"value": "42", "dimensions": {"concept": "ifrs-full:Assets", "entity": "scheme:1", "period": "2023-06-30T00:00:00", "unit": "iso4217:EUR"}},
            "dur": {"value": "7", "dimensions": {"concept": "ifrs-full:Revenue", "entity": "scheme:1", "period": "2022-01-01T00:00:00/2023-01-01T00:00:00"}}
          }
        }
        """
        result = _parser().parse(payload, accession="synth-1", entity_hint={})

        inst = next(f for f in result.facts if f.local_name == "Assets")
        assert inst.period_type is PeriodType.INSTANT
        assert inst.period_start is None
        assert inst.period_end == date(2023, 6, 30)
        assert inst.unit == "EUR"

        dur = next(f for f in result.facts if f.local_name == "Revenue")
        assert dur.period_type is PeriodType.DURATION
        assert dur.period_start == date(2022, 1, 1)
        assert dur.period_end == date(2023, 1, 1)
        assert dur.unit is None

    def test_boolean_values_are_not_treated_as_numeric(self):
        payload = b"""
        {
          "facts": {
            "b1": {"value": true, "dimensions": {"concept": "ifrs-full:SomeFlag", "entity": "scheme:1", "period": "2023-06-30T00:00:00"}}
          }
        }
        """
        result = _parser().parse(payload, accession="synth-bool", entity_hint={})

        assert result.facts == []
        assert any("non-numeric" in w for w in result.warnings)

    def test_pure_and_shares_units_are_mapped(self):
        payload = b"""
        {
          "facts": {
            "p": {"value": "0.5", "dimensions": {"concept": "ifrs-full:Ratio", "entity": "scheme:1", "period": "2023-01-01T00:00:00", "unit": "xbrli:pure"}},
            "s": {"value": "1000", "dimensions": {"concept": "ifrs-full:SharesOutstanding", "entity": "scheme:1", "period": "2023-01-01T00:00:00", "unit": "xbrli:shares"}}
          }
        }
        """
        result = _parser().parse(payload, accession="synth-units", entity_hint={})

        ratio = next(f for f in result.facts if f.local_name == "Ratio")
        shares = next(f for f in result.facts if f.local_name == "SharesOutstanding")
        assert ratio.unit == "pure"
        assert shares.unit == "shares"


class TestCanParse:
    def test_true_for_json_filename(self):
        parser = _parser()
        assert parser.can_parse(form_type=None, content_type=None, filename="43473797-2021-12-31.json") is True

    def test_true_for_json_content_type_regardless_of_filename(self):
        parser = _parser()
        assert parser.can_parse(form_type=None, content_type="application/json", filename="report") is True

    def test_false_for_zip_filename(self):
        parser = _parser()
        # ESEF ZIPs (inline XBRL + taxonomy package) are not handled by this
        # JSON-only parser -- a future Arelle-based parser owns that form.
        assert parser.can_parse(form_type=None, content_type=None, filename="report.zip") is False

    def test_false_for_unrelated_filename_and_content_type(self):
        parser = _parser()
        assert parser.can_parse(form_type="10-K", content_type="text/html", filename="report.htm") is False
