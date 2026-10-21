"""Chinese (Simplified) content, wave 3: loan/credit, deposits, remittance tracking (t22).

LLM-assisted, not native-speaker validated — see ``docs/ingestion.md`` §12.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank

EXTRA_INTENTS: dict[str, IntentBank] = {
    "loan_credit": IntentBank(
        title="个人消费贷款：申请、利率与提前还款",
        headings=("如何申请", "利率与审批条件", "提前还款"),
        sentences=(
            "贷款申请将在{days}个工作日内完成审批。",
            "贷款年利率为{percent}%，额度最高{limit} {currency}。",
            "审批需要证件{doc}和收入证明。",
            "审批结果会在{hours}小时内显示在应用和{email}邮箱中。",
            "贷款可在任意日期提前还款，不收取手续费。",
            "提前还款时利息按实际使用天数重新计算。",
            "还款计划可在应用和网点查询。",
        ),
        faqs=(
            (
                "如何提交贷款申请？",
                "可在应用或网点提交申请，审批时间为{days}个工作日。",
            ),
            (
                "可以提前还清贷款吗？",
                "可以，贷款可在任意日期提前还清，且不收取手续费。",
            ),
        ),
        table_rows=(
            "贷款利率|{percent}%|{days}天|{plan}",
            "贷款额度|{limit} {currency}|最长{days}天|—",
        ),
        procedure_steps=(
            "填写申请并上传证件{doc}。",
            "使用申请编号{order}查询审批结果。",
        ),
    ),
    "deposit_savings": IntentBank(
        title="存款与储蓄账户",
        headings=("如何开户", "利息与追加存款", "支取与销户"),
        sentences=(
            "存款可在应用或网点开立，期限从{days}天起。",
            "存款年利率为{percent}%。",
            "最低起存金额为{amount} {currency}。",
            "存款和定期存款可在线开立，资金当天到账。",
            "前{days}天内追加存款不受限制。",
            "利息按月计算并计入本金。",
            "到期前部分支取会将利率降至{percent}%。",
            "销户处理时间约为{hours}小时。",
        ),
        faqs=(
            (
                "如何开立存款？",
                "可在应用或网点开立，期限从{days}天起。",
            ),
            (
                "提前支取会怎样？",
                "到期前部分支取会将利率降至{percent}%。",
            ),
        ),
        table_rows=(
            "存款利率|{percent}%|{days}天|{plan}",
            "最低金额|{amount} {currency}|{hours}小时|—",
        ),
        procedure_steps=(
            "选择期限和金额。",
            "使用短信验证码确认开户。",
        ),
    ),
    "remittance_tracking": IntentBank(
        title="转账追踪：状态、查询、退款",
        headings=("转账状态", "款项未到账", "查询与退款"),
        sentences=(
            "转账状态可通过交易号{order}查询。",
            "向其他银行转账在{hours}小时内到账。",
            "向塔吉克斯坦转账通常在{hours}小时内到账。",
            "如果钱没有到亲属账户，请查询转账状态。",
            "若{days}天内未到账，可提交易查询申请。",
            "查询处理时间为{days}至{count}个工作日。",
            "收款银行确认后即可办理退款。",
            "如系银行差错，查询不收取手续费。",
            "查询进度会在应用和{phone}电话中更新。",
        ),
        faqs=(
            (
                "款项未到账怎么办？",
                "请用交易号{order}查询状态；若{days}天内仍未到账，请提交易查询申请。",
            ),
            (
                "转账查询需要多久？",
                "查询处理时间为{days}至{count}个工作日。",
            ),
        ),
        table_rows=(
            "转账到账|{hours}小时|{days}天|{channel}",
            "查询与退款|{days}天|{count}天|{channel}",
        ),
        procedure_steps=(
            "使用交易号{order}打开该笔交易。",
            "若款项未到账，提交易查询申请。",
        ),
    ),
}
