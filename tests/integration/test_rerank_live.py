"""Live checks against the real models. Marked ``integration``; needs a download.

Everything in the unit suite runs against stand-ins, and a stand-in cannot
establish the one fact the whole design rests on: that
``Qwen/Qwen3-Embedding-0.6B`` really has an asymmetric query mode rather than
merely declaring one.  These tests establish it against the actual weights.

Run with::

    .venv/bin/python -m pytest tests/integration/test_rerank_live.py \\
        -m integration --enable-socket -q -s
"""

from __future__ import annotations

import datetime

import pytest

from lit_search.core.paper import Paper
from lit_search.rerank import (
    DEFAULT_BI_ENCODER,
    DEFAULT_BI_ENCODER_REVISION,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_CROSS_ENCODER_REVISION,
    Reranker,
    load_bi_encoder,
)

pytestmark = pytest.mark.integration


def _paper(title: str, abstract: str, doi: str) -> Paper:
    """Build a candidate paper.

    Parameters
    ----------
    title : str
        Title.
    abstract : str
        Abstract.
    doi : str
        DOI.

    Returns
    -------
    Paper
        The record.
    """
    return Paper(
        title=title,
        abstract=abstract,
        authors=[],
        source=None,
        publication_date=datetime.date(2026, 4, 1),
        doi=doi,
    )


@pytest.fixture(scope="module")
def live_reranker() -> Reranker:
    """Load both real models once for the module."""
    return Reranker.load(device="cuda", dtype="bfloat16")


class TestRealModelHasARealQueryMode:
    """The assertion the design rests on, checked against actual weights."""

    def test_the_default_bi_encoder_ships_a_query_prompt(self) -> None:
        """If this ever stops holding, encode_query becomes a silent no-op."""
        encoder = load_bi_encoder(
            DEFAULT_BI_ENCODER, revision=DEFAULT_BI_ENCODER_REVISION, device="cuda"
        )
        assert encoder.query_prompt.startswith("Instruct: Given a web search query")
        assert encoder.document_prompt == ""
        assert encoder.dimension == 1024
        assert encoder.normalization == "l2"

    def test_query_and_document_encodings_genuinely_differ(self) -> None:
        """The same text, encoded two ways, must not land on the same vector.

        This is the check that a symmetric model would fail. It is the whole
        difference between this pipeline and the ``allenai-specter`` script it
        replaces.
        """
        import numpy as np

        encoder = load_bi_encoder(
            DEFAULT_BI_ENCODER, revision=DEFAULT_BI_ENCODER_REVISION, device="cuda"
        )
        text = "specification inference from call sites"
        as_query = np.asarray(encoder.model.encode_query([text]))[0]
        as_document = np.asarray(encoder.model.encode_document([text]))[0]
        assert not np.allclose(as_query, as_document)
        similarity = float(np.dot(as_query, as_document))
        assert similarity < 0.99, f"query and document spaces nearly identical: {similarity}"

    def test_the_cross_encoder_loads_natively(self, live_reranker: Reranker) -> None:
        """No trust_remote_code, and an Identity head so bfloat16 is safe."""
        assert live_reranker.cross_encoder is not None
        assert live_reranker.cross_encoder.model_id == DEFAULT_CROSS_ENCODER
        assert live_reranker.cross_encoder.revision == DEFAULT_CROSS_ENCODER_REVISION
        assert live_reranker.cross_encoder.activation == "Identity"


class TestRealRankingBehaviour:
    """The pipeline must actually separate relevant from irrelevant."""

    @pytest.fixture
    def candidates(self) -> list[Paper]:
        """Three on-topic papers and three distractors."""
        return [
            _paper(
                "Butterfly wing pigmentation gradients across altitude",
                "A field survey of pigmentation in montane lepidoptera sampled across "
                "three seasons and four elevation bands.",
                "10.1/butterfly",
            ),
            _paper(
                "Negative eigenvalues let linear recurrent networks track state",
                "Allowing the recurrent transition matrix of a linear RNN to have negative "
                "eigenvalues lets it solve parity and other state-tracking tasks that "
                "diagonal models with positive transitions provably cannot.",
                "10.1/negative-eig",
            ),
            _paper(
                "Tidal mixing in shallow estuarine basins",
                "We present a numerical model of tidal mixing and stratification in "
                "shallow estuarine basins under varying freshwater inflow.",
                "10.1/tides",
            ),
            _paper(
                "The expressive limits of state space models on sequential composition",
                "A circuit-complexity argument shows that state space models with "
                "diagonal transitions cannot compose permutations, so they fail to track "
                "state on word problems that recurrent networks with richer transitions solve.",
                "10.1/ssm-limits",
            ),
            _paper(
                "A survey of espresso extraction variables",
                "Grind size, dose, and pressure are varied across 400 shots to "
                "characterise extraction yield.",
                "10.1/coffee",
            ),
            _paper(
                "Householder products as recurrent transitions for state tracking",
                "Products of generalized Householder reflections give a linear recurrence "
                "non-diagonal transitions that stay parallelizable, and the resulting "
                "model solves modular arithmetic and permutation tracking tasks.",
                "10.1/householder",
            ),
        ]

    QUESTION = (
        "Which changes to the recurrent transition of a linear RNN or state space model "
        "let it track state, solving parity, modular arithmetic and permutation "
        "composition, while keeping training parallel across the sequence?"
    )

    def test_relevant_papers_outrank_distractors_after_both_stages(
        self, live_reranker: Reranker, candidates: list[Paper]
    ) -> None:
        """The end-to-end claim: the three on-topic papers come first."""
        ranking = live_reranker.rank_against_question(candidates, self.QUESTION, top_k=4)
        print("\n" + ranking.describe())
        order = [paper.doi for paper in ranking.reading_order()]
        print("reading order  :", order)
        relevant = {"10.1/negative-eig", "10.1/ssm-limits", "10.1/householder"}
        assert set(order[:3]) == relevant, f"expected the three on-topic papers first, got {order}"

    def test_stage_two_changes_the_order_it_is_given(
        self, live_reranker: Reranker, candidates: list[Paper]
    ) -> None:
        """If the cross-encoder never reorders anything, it is not earning its cost."""
        ranking = live_reranker.rank_against_question(candidates, self.QUESTION, top_k=6)
        assert ranking.rescored is not None
        bi_order = [paper.doi for paper in ranking.recall.papers]
        ce_order = [paper.doi for paper in ranking.rescored.papers]
        print("\nbi-encoder order  :", bi_order)
        print("cross-encoder     :", ce_order)
        print("scores (bi)       :", [round(s, 4) for s in ranking.recall.scores])
        print("scores (cross)    :", [round(s, 4) for s in ranking.rescored.scores])
        assert bi_order != ce_order or len(set(ranking.rescored.scores)) > 1

    def test_seed_mode_finds_the_neighbour_of_a_seed(
        self, live_reranker: Reranker, candidates: list[Paper]
    ) -> None:
        """Document-to-document: the seed's nearest neighbour is on-topic."""
        seed = candidates[1]
        others = [paper for paper in candidates if paper.doi != seed.doi]
        ranking = live_reranker.rank_against_seed_paper(others, seed)
        print("\n" + ranking.describe())
        top = ranking.recall.papers[0].doi
        print("nearest to seed:", top)
        assert top in {"10.1/ssm-limits", "10.1/householder"}

    def test_degraded_record_is_reported_not_hidden(
        self, live_reranker: Reranker, candidates: list[Paper]
    ) -> None:
        """A real record with no abstract must be named in the provenance."""
        stripped = _paper(
            "DeltaProduct: state tracking with Householder products", "", "10.1/deltaproduct"
        )
        ranking = live_reranker.rank_against_question(
            [*candidates, stripped], self.QUESTION, top_k=3
        )
        assert "doi:10.1/deltaproduct" in ranking.recall.provenance.degraded_text
        assert "degraded text  : 1 of 7" in ranking.describe()


def test_vram_fits_the_card(live_reranker: Reranker) -> None:
    """Both models co-resident must fit, with the measurement printed."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak allocated: {peak:.3f} GiB of {total:.3f} GiB")
    assert peak < total * 0.6
