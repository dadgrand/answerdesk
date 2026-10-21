"""Chinese (zh) content bank for the synthetic support corpus (FR-ING-10).

Simplified Chinese. Fictional products, reserved identifiers only (§13.4); no real
institution or brand is named. Every ``{slot}`` is provided by the generator.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank, LocaleBank

_INTENTS: dict[str, IntentBank] = {
    "tariffs": IntentBank(
        title="套餐与账户服务费",
        headings=("套餐概览", "月费与手续费", "更换套餐"),
        sentences=(
            "「{plan}」套餐每月{amount} {currency}，包含{count}次免费操作。",
            "首次开通「{plan}」套餐，第一个月免月费。",
            "月费在{date}扣取，账户余额不足时不会扣款。",
            "套餐每{days}天最多只能更换一次。",
            "「{plan}」套餐包含卡片服务和手机应用的全部功能。",
            "超出限额的操作手续费为{fee} {currency}。",
            "每日取现限额为{limit} {currency}。",
            "按年付费可享受{percent}%的折扣。",
        ),
        faqs=(
            (
                "「{plan}」套餐的服务费是多少？",
                "「{plan}」套餐的服务费为每月{amount} {currency}。",
            ),
            (
                "可以在应用中更换套餐吗？",
                "可以，在应用中即可更换，但每{days}天最多一次。",
            ),
        ),
        table_rows=(
            "月费|{amount} {currency}|{date}|{plan}",
            "超额手续费|{fee} {currency}|{days}天|{channel}",
        ),
        procedure_steps=(
            "在应用中打开「套餐」页面。",
            "选择「{plan}」套餐并点击「开通」。",
            "使用短信验证码确认变更。",
        ),
    ),
    "cards": IntentBank(
        title="银行卡：申请、限额与冻结",
        headings=("卡片申请", "卡片限额", "冻结与补卡"),
        sentences=(
            "申请通过后{days}个工作日内制卡并寄出。",
            "卡片日间限额为{limit} {currency}，夜间限额为{amount} {currency}。",
            "在他行取款机取现手续费为{fee} {currency}。",
            "卡片遗失时，请在应用中冻结并申请补卡。",
            "补卡费用为{amount} {currency}，再次补卡免费。",
            "卡号{card}的卡片支持网上支付和闪付。",
            "出现可疑交易时，卡片会自动冻结{hours}小时。",
            "密码可在取款机或应用中修改。",
        ),
        faqs=(
            (
                "补卡需要多少费用？",
                "补卡费用为{amount} {currency}，制卡需要{days}天。",
            ),
            (
                "卡片遗失后应该怎么办？",
                "请在应用中冻结卡片并申请补卡，通常需要{days}天。",
            ),
        ),
        table_rows=(
            "日间限额|{limit} {currency}|卡片|{plan}",
            "补卡费用|{amount} {currency}|{days}天|{channel}",
        ),
        procedure_steps=(
            "在应用中冻结卡片。",
            "申请补卡并填写收件地址。",
            "等待{days}个工作日收到新卡。",
        ),
    ),
    "transfers": IntentBank(
        title="转账：到账时间、手续费与限额",
        headings=("境内转账", "跨境转账", "状态与错误"),
        sentences=(
            "手机号转账实时到账，且不收取手续费。",
            "跨境转账需要{days}到{count}个工作日。",
            "跨境转账手续费为金额的{percent}%，最低{fee} {currency}。",
            "单笔转账最高金额为{limit} {currency}。",
            "如果款项未到账，请用单号{order}查询状态。",
            "错误码{error}表示收款人信息不存在或填写有误。",
            "转账只能在提交支付系统前撤销。",
            "每日转账限额为{limit} {currency}，可查询{count}个月的记录。",
        ),
        faqs=(
            (
                "跨境转账需要多长时间？",
                "跨境转账通常需要{days}到{count}个工作日。",
            ),
            (
                "错误码{error}是什么意思？",
                "错误码{error}表示收款人信息不存在或填写有误。",
            ),
        ),
        table_rows=(
            "境内转账|实时|{fee} {currency}|{channel}",
            "跨境转账|{days}天|{percent}%|{channel}",
        ),
        procedure_steps=(
            "核对收款人信息。",
            "确认金额与手续费。",
            "保存单号{order}以便查询。",
        ),
    ),
    "documents": IntentBank(
        title="证明文件：流水、对账单与补发",
        headings=("流水与对账单", "电子文件", "办理时间与费用"),
        sentences=(
            "账户流水证明需要{days}个工作日出具。",
            "电子对账单可在应用中查询最近{count}个月。",
            "纸质证明的费用为{amount} {currency}。",
            "证件{doc}只能在完成身份验证后申请。",
            "已办好的文件可在工作日{time}前到网点领取。",
            "文件的电子签名可在{site}核验。",
            "证件信息变更后，请在{days}天内更新资料。",
            "文件副本可在「文件」页面查看至{date}。",
        ),
        faqs=(
            (
                "如何获取账户流水证明？",
                "流水证明在{days}个工作日内出具，可在应用或网点申请。",
            ),
            (
                "纸质对账单需要多少费用？",
                "纸质对账单费用为{amount} {currency}，电子对账单免费。",
            ),
        ),
        table_rows=(
            "电子证明|免费|{days}天|{channel}",
            "纸质证明|{amount} {currency}|{days}天|网点",
        ),
        procedure_steps=(
            "在应用中打开「文件」页面。",
            "选择证件{doc}和领取方式。",
            "使用单号{order}查询办理状态。",
        ),
    ),
    "kyc": IntentBank(
        title="身份验证与资料确认",
        headings=("需要哪些资料", "审核时间", "驳回与重新提交"),
        sentences=(
            "身份验证通常需要{hours}小时，个别情况需要{days}天。",
            "需要证件{doc}照片、自拍照和地址证明。",
            "地址证明文件的开具时间不得超过{count}个月。",
            "资料不一致时，验证会以错误码{error}驳回。",
            "驳回后{hours}小时可以重新提交申请。",
            "资料以加密方式保存，未经同意不会提供给第三方。",
            "证件照片必须是彩色且没有反光。",
            "对驳回结果的申诉在{days}个工作日内处理。",
        ),
        faqs=(
            (
                "身份验证需要多长时间？",
                "验证通常需要{hours}小时，有时需要{days}天。",
            ),
            (
                "验证被驳回怎么办？",
                "请核对资料，并在{hours}小时后附上错误码{error}重新提交。",
            ),
        ),
        table_rows=(
            "身份验证|{hours}小时|{channel}|{doc}",
            "重新提交|{hours}小时|{channel}|—",
        ),
        procedure_steps=(
            "准备证件{doc}照片和自拍照。",
            "填写地址和联系方式。",
            "提交申请并等待审核结果。",
        ),
    ),
    "complaints": IntentBank(
        title="投诉与工单：流程与时限",
        headings=("如何提交工单", "处理时限", "升级与处理结果"),
        sentences=(
            "可以通过应用、{site}网站或电话{phone}提交工单。",
            "工单的处理时限为{days}个工作日。",
            "涉及错误码{error}的紧急工单在{hours}小时内处理。",
            "提交工单时请附上单号{order}和问题描述。",
            "工单状态在「帮助中心」页面更新。",
            "处理结果会发送到{email}并显示在应用中。",
            "如果结果不满意，可以联系高级客服。",
            "同一问题的重复工单会引用上一次结果后关闭。",
        ),
        faqs=(
            (
                "如何投诉手续费？",
                "请在应用或{site}网站提交工单，并附上单号{order}。",
            ),
            (
                "工单多久处理完成？",
                "通常需要{days}个工作日，紧急工单在{hours}小时内处理。",
            ),
        ),
        table_rows=(
            "普通工单|{days}天|{channel}|—",
            "紧急工单|{hours}小时|{channel}|{error}",
        ),
        procedure_steps=(
            "描述问题并附上单号{order}。",
            "选择沟通渠道：{channel}。",
            "在应用中跟踪工单状态。",
        ),
    ),
}

BANK: LocaleBank = LocaleBank(
    code="zh",
    display_name="Chinese",
    currency="CNY",
    contact_heading="联系方式与服务时间",
    contact_lines=(
        "客服电话：{phone}。",
        "邮箱：{email}。",
        "网站：{site}。每天{time}至20:00提供服务。",
        "更新日期：{date}。",
        "工单号：{order}。",
        "常见错误码：{error}。",
    ),
    faq_heading="常见问题",
    table_heading="套餐与限额",
    generic_headings=(
        "通用条款",
        "办理方式",
        "时限与状态",
        "安全提示",
        "下一步",
    ),
    generic_sentences=(
        "所有操作都可以在手机应用和{site}网站完成。",
        "遇到问题时，请拨打客服电话{phone}。",
        "我们在工作日{time}至20:00处理工单。",
        "可以使用单号{order}查询状态。",
        "变更自{date}起生效。",
        "处理结果会通过{email}通知客户。",
    ),
    procedure_heading="办理步骤",
    procedure_steps=(
        "打开应用并登录个人账户。",
        "选择对应页面并填写表单。",
        "使用短信验证码确认操作。",
        "保存工单号{order}。",
    ),
    table_header=("项目", "数值", "时限", "备注"),
    table_rows=(
        "手续费|{fee} {currency}|{days}天|{channel}",
        "限额|{limit} {currency}|每日|{plan}",
        "处理时限|{hours}小时|—|{channel}",
        "证明费用|{amount} {currency}|{days}天|{doc}",
        "费率|{percent}%|—|{channel}",
    ),
    plan_names=("入门版", "基础版", "舒适版", "尊享版"),
    channels=("手机应用", "网站", "电话", "网点"),
    document_names=("护照", "身份证", "驾驶证"),
    code_switch_pairs=(
        ("手续费", "fee"),
        ("转账", "transfer"),
        ("卡", "card"),
        ("限额", "limit"),
        ("时限", "deadline"),
        ("套餐", "plan"),
    ),
    date_templates=("自{date}起。", "截至{date}。", "（日期：{date}）"),
    intents=_INTENTS,
)
