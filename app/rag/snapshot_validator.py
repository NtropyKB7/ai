from app.schemas.finlife_product import FinlifePage, SnapshotCompleteness


def validate_complete_pages(pages: list[FinlifePage]) -> SnapshotCompleteness:
    if not pages:
        return SnapshotCompleteness(non_empty_and_not_abnormal_drop=False)
    page_numbers = sorted(page.now_page_no for page in pages)
    totals = {page.total_count for page in pages}
    maxima = {page.max_page_no for page in pages}
    expected = list(range(1, max(maxima) + 1)) if len(maxima) == 1 else []
    return SnapshotCompleteness(
        page_numbers_contiguous=page_numbers == expected,
        pagination_consistent=(
            len(totals) == 1
            and len(maxima) == 1
            and sum(len(page.bases) for page in pages) == next(iter(totals))
        ),
        non_empty_and_not_abnormal_drop=next(iter(totals), 0) > 0,
    )
