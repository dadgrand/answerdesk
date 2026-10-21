"""Uzbek (Latin) content, wave 3: loan/credit, deposits, remittance tracking (t22).

LLM-assisted, not native-speaker validated — see ``docs/ingestion.md`` §12.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank

EXTRA_INTENTS: dict[str, IntentBank] = {
    "loan_credit": IntentBank(
        title="Naqd kredit: ariza, stavka va muddatidan oldin to'lash",
        headings=("Ariza berish", "Stavka va tasdiqlash shartlari", "Muddatidan oldin to'lash"),
        sentences=(
            "Kredit arizasi {days} ish kuni ichida ko'rib chiqiladi.",
            "Kredit stavkasi {percent}% ni tashkil etadi, summa {limit} {currency} gacha.",
            "Tasdiqlash uchun {doc} hujjati va daromad tasdig'i kerak.",
            "Qaror {hours} soat ichida ilovada va {email} manzilida ko'rinadi.",
            "Kreditni istalgan kunda komissiyasiz muddatidan oldin to'lash mumkin.",
            "Muddatidan oldin to'lashda foizlar haqiqiy muddat uchun qayta hisoblanadi.",
            "To'lov grafigi ilovada va bo'limda mavjud.",
        ),
        faqs=(
            (
                "Kredit uchun qanday ariza beraman?",
                "Ariza ilovada yoki bo'limda beriladi va {days} ish kuni ichida ko'rib chiqiladi.",
            ),
            (
                "Kreditni muddatidan oldin to'lasa bo'ladimi?",
                "Ha, kreditni istalgan kunda komissiyasiz muddatidan oldin to'lash mumkin.",
            ),
        ),
        table_rows=(
            "Kredit stavkasi|{percent}%|{days} kun|{plan}",
            "Kredit summasi|{limit} {currency}|{days} kungacha|—",
        ),
        procedure_steps=(
            "Arizani to'ldiring va {doc} hujjatini qo'shing.",
            "{order} ariza raqami bo'yicha qarorni kuting.",
        ),
    ),
    "deposit_savings": IntentBank(
        title="Omonatlar va jamg'arma hisoblari",
        headings=("Omonat ochish", "Foiz va to'ldirish", "Yechish va yopish"),
        sentences=(
            "Omonat ilovada yoki bo'limda {days} kundan boshlab ochiladi.",
            "Omonat foizi {percent}% ni tashkil etadi.",
            "Minimal summa — {amount} {currency}.",
            "Depozit va omonatni onlayn ochish mumkin, pul shu kuni tushadi.",
            "Birinchi {days} kun ichida to'ldirish cheklanmagan.",
            "Foizlar har oy hisoblanadi va kapitallashtiriladi.",
            "Muddatdan oldin qismini yechish stavkani {percent}% gacha pasaytiradi.",
            "Omonatni yopish {hours} soatgacha davom etadi.",
        ),
        faqs=(
            (
                "Omonatni qanday ochaman?",
                "Omonat ilovada yoki bo'limda {days} kundan boshlab ochiladi.",
            ),
            (
                "Muddatdan oldin yechsam nima bo'ladi?",
                "Muddatdan oldin qismini yechish stavkani {percent}% gacha pasaytiradi.",
            ),
        ),
        table_rows=(
            "Omonat foizi|{percent}%|{days} kun|{plan}",
            "Minimal summa|{amount} {currency}|{hours} soat|—",
        ),
        procedure_steps=(
            "Muddat va summani tanlang.",
            "Ochishni SMS kod bilan tasdiqlang.",
        ),
    ),
    "remittance_tracking": IntentBank(
        title="O'tkazmani kuzatish: holat, qidiruv, qaytarish",
        headings=("O'tkazma holati", "Pul kelmadi", "Qidiruv va qaytarish"),
        sentences=(
            "O'tkazma holati {order} amaliyot raqami bo'yicha tekshiriladi.",
            "Boshqa bankka o'tkazma {hours} soat ichida tushadi.",
            "Tojikistonga o'tkazma odatda {hours} soat ichida tushadi.",
            "Pul qarindoshga yetib bormasa, o'tkazma holatini tekshiring.",
            "Pul {days} kun ichida kelmasa, qidiruv arizasi beriladi.",
            "Qidiruv {days} kundan {count} ish kunigacha davom etadi.",
            "Pul qabul qiluvchi bank tasdiqlagandan keyin qaytariladi.",
            "Bank xatosida qidiruv uchun komissiya olinmaydi.",
            "Qidiruv holati ilovada va {phone} raqami orqali yangilanadi.",
        ),
        faqs=(
            (
                "Pul kelmadi, nima qilaman?",
                "O'tkazma holatini {order} raqami bo'yicha tekshiring; {days} kun ichida kelmasa, "
                "qidiruv arizasini bering.",
            ),
            (
                "O'tkazma qidiruvi qancha vaqt oladi?",
                "Qidiruv {days} kundan {count} ish kunigacha davom etadi.",
            ),
        ),
        table_rows=(
            "O'tkazma tushishi|{hours} soat|{days} kun|{channel}",
            "Qidiruv va qaytarish|{days} kun|{count} kun|{channel}",
        ),
        procedure_steps=(
            "{order} raqami bo'yicha amaliyotni oching.",
            "Pul kelmasa, qidiruv arizasini bering.",
        ),
    ),
}
