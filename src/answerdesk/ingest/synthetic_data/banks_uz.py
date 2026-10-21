"""Uzbek (uz, Latin script) content bank for the synthetic support corpus (FR-ING-10).

Fictional products, reserved identifiers only (§13.4); no real institution or brand is
named. Every ``{slot}`` is provided by the generator.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank, LocaleBank

_INTENTS: dict[str, IntentBank] = {
    "tariffs": IntentBank(
        title="Tariflar va xizmat narxi",
        headings=("Tariflar haqida", "Oylik to'lov va komissiya", "Tarifni almashtirish"),
        sentences=(
            "«{plan}» tarifi oyiga {amount} {currency} turadi va {count} bepul "
            "amaliyotni o'z ichiga "
            "oladi.",
            "«{plan}» tarifiga ulanganda birinchi oy bepul.",
            "Oylik to'lov {date} sanasida yechiladi, agar hisobda mablag' yetarli bo'lsa.",
            "Tarifni {days} kunda bir martadan ko'p almashtirish mumkin emas.",
            "«{plan}» tarifiga karta xizmati va mobil ilova kiradi.",
            "Limitdan oshgan amaliyot uchun komissiya {fee} {currency}.",
            "Naqd pul yechish limiti sutkasiga {limit} {currency}.",
            "Yillik to'lovda {percent}% chegirma mavjud.",
        ),
        faqs=(
            (
                "«{plan}» tarifi bo'yicha xizmat qancha turadi?",
                "«{plan}» tarifi oyiga {amount} {currency} turadi.",
            ),
            (
                "Tarifni ilovada almashtirsa bo'ladimi?",
                "Ha, tarifni ilovada almashtirish mumkin, lekin {days} kunda bir martadan ko'p "
                "emas.",
            ),
        ),
        table_rows=(
            "Oylik to'lov|{amount} {currency}|{date}|{plan}",
            "Ortiqcha komissiya|{fee} {currency}|{days} kun|{channel}",
        ),
        procedure_steps=(
            "Ilovada «Tariflar» bo'limini oching.",
            "«{plan}» tarifini tanlang va «Ulanish» tugmasini bosing.",
            "O'zgartirishni SMS kod bilan tasdiqlang.",
        ),
    ),
    "cards": IntentBank(
        title="Kartalar: chiqarish, limitlar va bloklash",
        headings=("Kartani chiqarish", "Karta limitlari", "Bloklash va qayta chiqarish"),
        sentences=(
            "Karta tasdiqlangandan so'ng {days} ish kuni ichida chiqariladi.",
            "Kartaning kunlik limiti {limit} {currency}, tungi limiti {amount} {currency}.",
            "Boshqa bankomatlarda naqd yechish komissiyasi {fee} {currency}.",
            "Karta yo'qolsa, uni ilovada bloklang va qayta chiqarishga buyurtma bering.",
            "Kartani qayta chiqarish {amount} {currency} turadi.",
            "{card} raqamli karta uchun internet to'lovlari mavjud.",
            "Shubhali amaliyotda karta {hours} soatgacha avtomatik bloklanadi.",
            "PIN kodni bankomatda yoki ilovada o'zgartirish mumkin.",
        ),
        faqs=(
            (
                "Kartani qayta chiqarish qancha turadi?",
                "Qayta chiqarish {amount} {currency} turadi va {days} kun davom etadi.",
            ),
            (
                "Karta yo'qolsa nima qilish kerak?",
                "Kartani ilovada bloklang va qayta chiqarishga buyurtma bering — bu {days} kun "
                "oladi.",
            ),
        ),
        table_rows=(
            "Kunlik limit|{limit} {currency}|karta|{plan}",
            "Qayta chiqarish|{amount} {currency}|{days} kun|{channel}",
        ),
        procedure_steps=(
            "Kartani ilovada bloklang.",
            "Qayta chiqarishga buyurtma bering va yetkazish manzilini kiriting.",
            "Kartani {days} ish kuni ichida kuting.",
        ),
    ),
    "transfers": IntentBank(
        title="O'tkazmalar: muddat, komissiya va limitlar",
        headings=("Ichki o'tkazmalar", "Xalqaro o'tkazmalar", "Holatlar va xatolar"),
        sentences=(
            "Telefon raqami bo'yicha o'tkazma darhol va komissiyasiz amalga oshiriladi.",
            "Xalqaro o'tkazma {days} kundan {count} ish kunigacha davom etadi.",
            "Xalqaro o'tkazma komissiyasi summaning {percent}%, kamida {fee} {currency}.",
            "Bitta o'tkazmaning maksimal summasi {limit} {currency}.",
            "O'tkazma kelmasa, {order} raqami bo'yicha holatni tekshiring.",
            "{error} xatosi qabul qiluvchi topilmaganini bildiradi.",
            "O'tkazmani faqat to'lov tizimiga yuborilgunga qadar bekor qilish mumkin.",
            "Sutkalik limit {limit} {currency}; tarix {count} oy uchun mavjud.",
        ),
        faqs=(
            (
                "Xalqaro o'tkazma qancha vaqt oladi?",
                "Xalqaro o'tkazma {days} kundan {count} ish kunigacha davom etadi.",
            ),
            (
                "{error} xatosi nimani bildiradi?",
                "{error} xatosi qabul qiluvchi topilmaganini yoki rekvizitlar xato ekanini "
                "bildiradi.",
            ),
        ),
        table_rows=(
            "Ichki o'tkazma|darhol|{fee} {currency}|{channel}",
            "Xalqaro o'tkazma|{days} kun|{percent}%|{channel}",
        ),
        procedure_steps=(
            "Qabul qiluvchining rekvizitlarini tekshiring.",
            "Summa va komissiyani tasdiqlang.",
            "Kuzatish uchun {order} amaliyot raqamini saqlang.",
        ),
    ),
    "documents": IntentBank(
        title="Hujjatlar: ma'lumotnoma, ko'chirma va tiklash",
        headings=("Ma'lumotnoma va ko'chirma", "Elektron hujjatlar", "Muddat va narx"),
        sentences=(
            "Mablag' harakati to'g'risidagi ma'lumotnoma {days} ish kunida tayyorlanadi.",
            "Elektron ko'chirma ilovada oxirgi {count} oy uchun mavjud.",
            "Qog'oz ma'lumotnoma narxi {amount} {currency}.",
            "{doc} hujjatini faqat shaxs tasdiqlangandan keyin buyurtma qilish mumkin.",
            "Tayyor hujjatni bo'limda ish kunlari soat {time} gacha olish mumkin.",
            "Hujjatning elektron imzosi {site} saytida tekshiriladi.",
            "Pasport ma'lumotlari o'zgarsa, hujjatlarni {days} kun ichida yangilang.",
            "Hujjat nusxasi «Hujjatlar» bo'limida {date} sanasigacha mavjud.",
        ),
        faqs=(
            (
                "Mablag' harakati to'g'risidagi ma'lumotnomani qanday olaman?",
                "Ma'lumotnoma {days} ish kunida tayyorlanadi; uni ilovada yoki bo'limda buyurtma "
                "qiling.",
            ),
            (
                "Qog'oz ko'chirma qancha turadi?",
                "Qog'oz ko'chirma {amount} {currency}, elektron ko'chirma bepul.",
            ),
        ),
        table_rows=(
            "Elektron ma'lumotnoma|bepul|{days} kun|{channel}",
            "Qog'oz ma'lumotnoma|{amount} {currency}|{days} kun|bo'lim",
        ),
        procedure_steps=(
            "Ilovada «Hujjatlar» bo'limini oching.",
            "{doc} hujjatini va olish usulini tanlang.",
            "Buyurtma holatini {order} raqami bo'yicha tekshiring.",
        ),
    ),
    "kyc": IntentBank(
        title="Shaxsni tasdiqlash va ma'lumotlarni tekshirish",
        headings=("Qanday ma'lumot kerak", "Tekshirish muddati", "Rad etish va qayta yuborish"),
        sentences=(
            "Shaxsni tasdiqlash {hours} soatgacha, kamdan-kam hollarda {days} kun davom etadi.",
            "{doc} hujjati surati, selfi va manzil tasdig'i kerak.",
            "Manzil {count} oydan eski bo'lmagan hujjat bilan tasdiqlanadi.",
            "Ma'lumotlar mos kelmasa, tekshirish {error} kodi bilan rad etiladi.",
            "Qayta yuborish rad etishdan {hours} soat keyin mumkin.",
            "Ma'lumotlar shifrlangan holda saqlanadi va roziliksiz uchinchi shaxsga berilmaydi.",
            "Hujjat surati rangli va yorqin bo'lishi kerak.",
            "Rad etish bo'yicha shikoyat {days} ish kunida ko'rib chiqiladi.",
        ),
        faqs=(
            (
                "Shaxsni tasdiqlash qancha vaqt oladi?",
                "Tekshirish {hours} soatgacha, ba'zan {days} kungacha davom etadi.",
            ),
            (
                "Rad etilganda nima qilish kerak?",
                "Ma'lumotlarni tekshiring va {hours} soatdan keyin {error} kodini ko'rsatib qayta "
                "yuboring.",
            ),
        ),
        table_rows=(
            "Shaxsni tasdiqlash|{hours} soat|{channel}|{doc}",
            "Qayta yuborish|{hours} soat|{channel}|—",
        ),
        procedure_steps=(
            "{doc} hujjati surati va selfini tayyorlang.",
            "Manzil va aloqa ma'lumotlarini to'ldiring.",
            "Arizani yuboring va natijani kuting.",
        ),
    ),
    "complaints": IntentBank(
        title="Murojaatlar va shikoyatlar: tartib va muddat",
        headings=(
            "Murojaat qanday yuboriladi",
            "Ko'rib chiqish muddati",
            "Eskalatsiya va qarorlar",
        ),
        sentences=(
            "Murojaatni ilovada, {site} saytida yoki {phone} raqami orqali yuborish mumkin.",
            "Murojaatni ko'rib chiqish muddati {days} ish kunigacha.",
            "{error} xatosiga oid shoshilinch murojaatlar {hours} soatda ko'rib chiqiladi.",
            "Murojaatga {order} amaliyot raqamini va muammo tavsifini qo'shing.",
            "Murojaat holati «Qo'llab-quvvatlash» bo'limida yangilanadi.",
            "Qaror {email} manziliga va ilovaga yuboriladi.",
            "Qaror mos kelmasa, katta operatorga murojaat qiling.",
            "Xuddi shu mavzudagi takroriy murojaat oldingi qarorga havola bilan yopiladi.",
        ),
        faqs=(
            (
                "Komissiya bo'yicha shikoyatni qanday yuboraman?",
                "Murojaatni ilovada yoki {site} saytida yuboring va {order} amaliyot raqamini "
                "qo'shing.",
            ),
            (
                "Murojaat qancha vaqtda ko'rib chiqiladi?",
                "Odatda {days} ish kunigacha; shoshilinch murojaatlar — {hours} soatgacha.",
            ),
        ),
        table_rows=(
            "Oddiy murojaat|{days} kun|{channel}|—",
            "Shoshilinch murojaat|{hours} soat|{channel}|{error}",
        ),
        procedure_steps=(
            "Muammoni tavsiflang va {order} amaliyot raqamini qo'shing.",
            "Aloqa kanalini tanlang: {channel}.",
            "Murojaat holatini ilovada kuzatib boring.",
        ),
    ),
}

BANK: LocaleBank = LocaleBank(
    code="uz",
    display_name="Uzbek",
    currency="UZS",
    contact_heading="Aloqa va ish vaqti",
    contact_lines=(
        "Qo'llab-quvvatlash telefoni: {phone}.",
        "Pochta: {email}.",
        "Sayt: {site}. Har kuni {time} dan 20:00 gacha ishlaymiz.",
        "{date} holatiga ko'ra amal qiladi.",
        "Murojaat raqami: {order}.",
        "Odatiy xato kodi: {error}.",
    ),
    faq_heading="Ko'p beriladigan savollar",
    table_heading="Tariflar va limitlar",
    generic_headings=(
        "Umumiy shartlar",
        "Bu qanday ishlaydi",
        "Muddat va holat",
        "Xavfsizlik",
        "Keyingi qadam",
    ),
    generic_sentences=(
        "Barcha amaliyotlar mobil ilovada va {site} saytida mavjud.",
        "Muammo bo'lsa, {phone} raqami orqali qo'llab-quvvatlashga murojaat qiling.",
        "Murojaatlarni ish kunlari {time} dan 20:00 gacha ko'rib chiqamiz.",
        "Holatni {order} raqami bo'yicha tekshirish mumkin.",
        "O'zgarishlar {date} sanasidan kuchga kiradi.",
        "Mijoz qaror haqida {email} manziliga xabar oladi.",
    ),
    procedure_heading="Amallar tartibi",
    procedure_steps=(
        "Ilovani oching va shaxsiy kabinetga kiring.",
        "Kerakli bo'limni tanlang va shaklni to'ldiring.",
        "Amalni SMS kod bilan tasdiqlang.",
        "Murojaat raqamini saqlang: {order}.",
    ),
    table_header=("Ko'rsatkich", "Qiymat", "Muddat", "Izoh"),
    table_rows=(
        "Komissiya|{fee} {currency}|{days} kun|{channel}",
        "Limit|{limit} {currency}|sutka|{plan}",
        "Ishlov muddati|{hours} soat|—|{channel}",
        "Ma'lumotnoma narxi|{amount} {currency}|{days} kun|{doc}",
        "Komissiya ulushi|{percent}%|—|{channel}",
    ),
    plan_names=("Boshlang'ich", "Asosiy", "Qulay", "Premium"),
    channels=("ilova", "sayt", "telefon", "bo'lim"),
    document_names=("pasport", "ID karta", "haydovchilik guvohnomasi"),
    code_switch_pairs=(
        ("komissiya", "fee"),
        ("o'tkazma", "transfer"),
        ("karta", "card"),
        ("chegara", "limit"),
        ("muddat", "deadline"),
        ("tarif", "plan"),
    ),
    date_templates=("{date} dan.", "{date} gacha.", "(sana: {date})"),
    intents=_INTENTS,
)
