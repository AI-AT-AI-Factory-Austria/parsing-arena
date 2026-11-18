import pytest

from app.engines.tools import DoclingRuntime


def test_export_preserves_page_text_without_image_classifier_labels():
    doc_types = pytest.importorskip("docling_core.types.doc")
    doc = doc_types.DoclingDocument(name="Export regression")
    for number in (1, 2):
        doc.add_page(number, doc_types.Size(width=100, height=100))

    def provenance(page, y):
        return doc_types.ProvenanceItem(
            page_no=page,
            charspan=(0, 10),
            bbox=doc_types.BoundingBox(
                l=5, t=y, r=95, b=y + 5, coord_origin=doc_types.CoordOrigin.TOPLEFT,
            ),
        )

    for number in (1, 2):
        doc.add_text(
            label=doc_types.DocItemLabel.PAGE_HEADER,
            text=f"Header {number}",
            content_layer=doc_types.ContentLayer.FURNITURE,
            prov=provenance(number, 5),
        )
        doc.add_text(
            label=doc_types.DocItemLabel.PAGE_FOOTER,
            text=f"Footer {number}",
            content_layer=doc_types.ContentLayer.FURNITURE,
            prov=provenance(number, 90),
        )
    picture = doc.add_picture(prov=provenance(1, 20))
    picture.meta = doc_types.PictureMeta.model_validate({
        "classification": {"predictions": [{
            "class_name": "logo", "confidence": 1.0, "created_by": "test",
        }]},
    })
    doc.add_text(
        label=doc_types.DocItemLabel.TEXT,
        text="Recognized text inside picture",
        parent=picture,
        prov=provenance(1, 25),
    )
    doc.add_text(
        label=doc_types.DocItemLabel.TEXT,
        text="Logo policy: Other uses",
        prov=provenance(1, 40),
    )

    default = doc.export_to_markdown(page_no=1)
    assert "\n\nLogo\n\n" in default
    assert "Header 1" not in default and "Recognized text inside picture" not in default

    markdown = DoclingRuntime._markdown(doc, 1)
    for text in (
        "Header 1", "Footer 1", "Recognized text inside picture", "Logo policy: Other uses",
    ):
        assert text in markdown
    assert "\n\nLogo\n\n" not in markdown
    assert "Header 2" not in markdown and "Footer 2" not in markdown
    elements = DoclingRuntime._elements(doc, 1)
    assert [item["content"] for item in elements] == [
        "Header 1", "Footer 1", "", "Recognized text inside picture", "Logo policy: Other uses",
    ]
    assert elements[0]["category"] == "Page-header"
    assert elements[1]["category"] == "Page-footer"
    assert elements[3]["bbox"] == [0.05, 0.25, 0.95, 0.3]
