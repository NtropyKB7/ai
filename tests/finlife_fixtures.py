from copy import deepcopy


def response(product_type="SAVINGS", *, option=True, total="1", page="1", max_page="1"):
    is_savings = product_type == "SAVINGS"
    base = {
        "dcls_month": "202608",
        "fin_co_no": "0010001",
        "fin_prdt_cd": "P001",
        "kor_co_nm": "합성은행",
        "fin_prdt_nm": "합성 적금" if is_savings else "합성 예금",
        "join_way": "인터넷",
        "mtrt_int": "원천 만기 문구",
        "spcl_cnd": "원천 우대조건",
        "join_deny": "1",
        "join_member": "제한없음",
        "etc_note": "합성 fixture",
        "max_limit": None,
        "dcls_strt_day": "20260801",
        "dcls_end_day": None,
        "fin_co_subm_day": "202608011200",
    }
    option_item = {
        "dcls_month": "202608",
        "fin_co_no": "0010001",
        "fin_prdt_cd": "P001",
        "intr_rate_type": "S",
        "intr_rate_type_nm": "단리",
        "save_trm": "12",
        "intr_rate": None,
        "intr_rate2": "3.50",
    }
    if is_savings:
        option_item.update(rsrv_type="F", rsrv_type_nm="자유적립식")
    return {
        "result": {
            "prdt_div": "S" if is_savings else "D",
            "total_count": total,
            "max_page_no": max_page,
            "now_page_no": page,
            "err_cd": "000",
            "err_msg": "정상",
            "baseList": [base],
            "optionList": [option_item] if option else [],
        }
    }


def changed(payload, path, value):
    result = deepcopy(payload)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return result
