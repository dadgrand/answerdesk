"""Uzbek (Latin) content, wave 2: FAQ, fee table, app guide, fraud policy, closure, security.

LLM-assisted, not native-speaker validated — see ``docs/ingestion.md`` §12.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank

EXTRA_INTENTS: dict[str, IntentBank] = {
    "faq": IntentBank(
        title="Kartalar va o'tkazmalar bo'yicha ko'p beriladigan savollar",
        headings=("Umumiy savollar", "To'lov va limitlar", "Hujjatlar va muddatlar"),
        sentences=(
            "To'lov o'tmasa, limit va karta amal qilish muddatini tekshiring.",
            "Telefon raqami bo'yicha o'tkazma qabul qiluvchiga darhol tushadi.",
            "Xizmat haqi {date} sanasida yechiladi va amaliyotlar soniga bog'liq emas.",
            "Sutkalik o'tkazma limiti {limit} {currency}.",
            "Telefon raqamini o'zgartirish uchun {doc} hujjati va SMS tasdiq kerak.",
            "Ariza holati {hours} soat ichida ilovada yangilanadi.",
            "Mablag' harakati to'g'risidagi ma'lumotnomani «Hujjatlar» bo'limida buyurtma qiling.",
            "Agar amaliyot {error} kodi bilan rad etilsa, ma'lumotlarni tekshirib qayta yuboring.",
            "Xato yechilgan summa {days} ish kuni ichida qaytariladi.",
        ),
        faqs=(
            (
                "To'lov nima uchun rad etildi?",
                "To'lov {limit} {currency} limitidan oshsa yoki karta bloklangan bo'lsa rad "
                "etiladi.",
            ),
            (
                "Telefon raqami bo'yicha o'tkazma qancha vaqtda keladi?",
                "Telefon raqami bo'yicha o'tkazma qabul qiluvchiga darhol tushadi.",
            ),
        ),
        table_rows=(
            "Murojaatga javob|{days} kun|{channel}|{plan}",
            "O'tkazma muddati|{hours} soat|{channel}|—",
        ),
        procedure_steps=(
            "«Qo'llab-quvvatlash» bo'limini ochib, mavzuni tanlang.",
            "Vaziyatni yozib, {order} amaliyot raqamini qo'shing.",
        ),
    ),
    "fee_table": IntentBank(
        title="To'lovlar va komissiyalar: to'liq jadval",
        headings=("To'lovlar ro'yxati", "Komissiya olinmaydigan holatlar", "Komissiyani qaytarish"),
        sentences=(
            "Boshqa bankka o'tkazma komissiyasi {fee} {currency}.",
            "O'z bankomatlarimizda naqd pul yechish uchun komissiya olinmaydi.",
            "Karta xizmati «{plan}» tarifiga kiradi va alohida to'lanmaydi.",
            "Shoshilinch o'tkazma uchun {percent}% komissiya, kamida {fee} {currency}.",
            "Xalqaro o'tkazma komissiyasi summa va valyutaga bog'liq.",
            "Internetda to'lov qilishda bank komissiyasi olinmaydi.",
            "Komissiya tariflari {days} kunda bir martadan ko'p qayta ko'rilmaydi.",
            "Xato yechilgan komissiya {days} kun ichida qaytariladi.",
            "To'lovlarning to'liq ro'yxati ilovada va {site} saytida mavjud.",
        ),
        faqs=(
            (
                "Boshqa bankka o'tkazma qancha turadi?",
                "Boshqa bankka o'tkazma komissiyasi {fee} {currency}.",
            ),
            (
                "Yechilgan komissiyani qaytarish mumkinmi?",
                "Xato yechilgan komissiya {days} kun ichida hisobga qaytariladi.",
            ),
        ),
        table_rows=(
            "Boshqa bankka o'tkazma|{fee} {currency}|{days} kun|{channel}",
            "Shoshilinch o'tkazma|{percent}%|darhol|{channel}",
        ),
        procedure_steps=(
            "Ko'chirmadan komissiya olingan amaliyotni toping.",
            "{order} amaliyot raqami bilan murojaat yuboring.",
        ),
    ),
    "app_guide": IntentBank(
        title="Mobil ilova: bosqichma-bosqich qo'llanma",
        headings=("Kirish va sozlash", "To'lovlar va o'tkazmalar", "Ilovada xavfsizlik"),
        sentences=(
            "Ilovaga kirish SMS kod yoki biometrika bilan tasdiqlanadi.",
            "Qabul qiluvchi qo'shish uchun «O'tkazmalar» bo'limini oching.",
            "Amaliyotlar tarixini sana, summa va to'lov turi bo'yicha filtrash mumkin.",
            "Yechimlar haqidagi xabarnomalar profil sozlamalarida yoqiladi.",
            "To'lov shabloni birinchi to'lovdan keyin saqlanadi.",
            "Ilovani yangilash zaif tarmoqda {hours} soatgacha davom etadi.",
            "Qurilmani almashtirsangiz, {phone} raqamini qayta tasdiqlang.",
            "Ilova ochilmasa, versiya va bo'sh joyni tekshiring.",
            "Qo'llab-quvvatlash chati ilovada kechayu kunduz ishlaydi.",
        ),
        faqs=(
            (
                "O'tkazma uchun qabul qiluvchini qanday qo'shaman?",
                "«O'tkazmalar» bo'limini ochib, «Yangi qabul qiluvchi» tugmasini bosing.",
            ),
            (
                "Ilova ochilmasa nima qilish kerak?",
                "Ilova versiyasi va bo'sh joyni tekshirib, qaytadan kiring.",
            ),
        ),
        table_rows=(
            "Biometrika bilan kirish|mavjud|{plan}|{channel}",
            "Ilovani yangilash|{hours} soat|{channel}|—",
        ),
        procedure_steps=(
            "Ilovani ochib, profilga kiring.",
            "«O'tkazmalar» bo'limini tanlab, shaklni to'ldiring.",
        ),
    ),
    "fraud_policy": IntentBank(
        title="Firibgarlikka qarshi kurash siyosati",
        headings=("Firibgarlik belgilari", "Bank nima qiladi", "Mijoz nima qilishi kerak"),
        sentences=(
            "Bank hech qachon telefon orqali parol yoki SMS kodni so'ramaydi.",
            "Shubhali amaliyotlar {hours} soatgacha avtomatik bloklanadi.",
            "Agar mijoz SMS kodni aytib qo'ysa, kartani darhol bloklash kerak.",
            "Mablag'ni qaytarish arizasi {days} ish kuni ichida ko'rib chiqiladi.",
            "Qo'ng'iroq qilgan raqam qayd etiladi va xavfsizlik xizmatiga yuboriladi.",
            "Firibgarga o'tkazilgan pulni {days} kun ichida e'tiroz qilish mumkin.",
            "Ilovaga kirishlar haqida xabar {email} manziliga keladi.",
            "Xodimni faqat {site} saytida ko'rsatilgan {phone} raqami orqali tekshiring.",
            "Siyosat har yili yangi firibgarlik sxemalarini hisobga olib qayta ko'riladi.",
        ),
        faqs=(
            (
                "Bank SMS kodni so'rashi mumkinmi?",
                "Yo'q, bank hech qachon telefon orqali parol yoki SMS kodni so'ramaydi.",
            ),
            (
                "Kodni firibgarga aytib qo'ysam nima qilaman?",
                "Kartani darhol bloklab, {phone} raqamiga qo'ng'iroq qiling.",
            ),
        ),
        table_rows=(
            "Shubhali amaliyotni bloklash|{hours} soat|{channel}|{plan}",
            "Mablag'ni qaytarish|{days} kun|{channel}|—",
        ),
        procedure_steps=(
            "Kartani ilovada bloklang.",
            "{phone} raqamiga qo'ng'iroq qilib, {order} amaliyot kodini ayting.",
        ),
    ),
    "account_closure": IntentBank(
        title="Hisobni yopish va shartnomani bekor qilish",
        headings=("Hisobni qachon yopish mumkin", "Yopish tartibi", "Keyin nima bo'ladi"),
        sentences=(
            "Hisobni barcha amaliyotlar tugagandan keyin yopish mumkin.",
            "Yopish arizasi bo'limda yoki ilovada topshiriladi.",
            "Hisobni yopish muddati {days} ish kunigacha.",
            "Qoldiq mablag' ko'rsatilgan rekvizitlarga o'tkaziladi yoki naqd beriladi.",
            "Xizmat bo'yicha qarz yopishda qoldiqdan ushlab qolinadi.",
            "Hisob yopilgach shartnoma bekor qilinadi va karta bloklanadi.",
            "Yopilganlik to'g'risidagi ma'lumotnoma {days} kun ichida beriladi.",
            "Hisobni qayta ochish yopilgandan keyin shu kuni mumkin.",
            "Hisobni yopish avval berilgan kredit majburiyatlarini bekor qilmaydi.",
        ),
        faqs=(
            (
                "Hisobni yopish qancha vaqt oladi?",
                "Ariza topshirilgandan keyin hisob {days} ish kuni ichida yopiladi.",
            ),
            (
                "Yopishda qoldiq qayerga o'tkaziladi?",
                "Qoldiq ko'rsatilgan rekvizitlarga o'tkaziladi yoki naqd beriladi.",
            ),
        ),
        table_rows=(
            "Hisobni yopish|{days} kun|{channel}|{plan}",
            "Yopilganlik ma'lumotnomasi|{days} kun|{channel}|—",
        ),
        procedure_steps=(
            "Hisobdagi barcha amaliyotlar tugaganini tekshiring.",
            "Yopish arizasini topshirib, qoldiq uchun rekvizitlarni ko'rsating.",
        ),
    ),
    "security_notice": IntentBank(
        title="Xavfsizlik: kodlar, parollar va shubhali qo'ng'iroqlar",
        headings=("Xavfsizlik qoidalari", "Kodlar va parollar", "Qayerga xabar berish"),
        sentences=(
            "SMS kodni hech kimga aytmang, hatto o'zini bank xodimi deb tanishtirsa ham.",
            "Ilova paroli faqat mijozda saqlanadi va qo'llab-quvvatlash xizmati so'ramaydi.",
            "Yangi qurilmadan kirish qo'shimcha tasdiqlanadi.",
            "Shubhali qo'ng'iroq bo'lsa, suhbatni tugatib {phone} raqamiga qo'ng'iroq qiling.",
            "Xodimni {site} saytida ko'rsatilgan {phone} raqami orqali tekshirish mumkin.",
            "Shubhali amaliyotlar haqida {email} manziliga xabar yuboring.",
            "SMS kod {hours} soat amal qiladi va bir marta ishlatiladi.",
            "Uch marta xato kirishdan keyin ilova {hours} soatga bloklanadi.",
            "Xavfsizlik bo'yicha ma'lumot {site} saytining «Yordam» bo'limida.",
        ),
        faqs=(
            (
                "Bank xodimi SMS kodni so'rasa, bu normalmi?",
                "Yo'q, SMS kodni hech kimga, hatto bank xodimiga ham aytmaslik kerak.",
            ),
            (
                "Shubhali qo'ng'iroq haqida qayerga xabar beraman?",
                "Suhbatni tugatib, {site} saytida ko'rsatilgan {phone} raqamiga qo'ng'iroq qiling.",
            ),
        ),
        table_rows=(
            "Kod amal qilish muddati|{hours} soat|{channel}|{plan}",
            "Xatodan keyin bloklash|{hours} soat|{channel}|—",
        ),
        procedure_steps=(
            "Shubhali suhbatni tugating.",
            "{phone} raqamiga qo'ng'iroq qilib, vaziyatni tushuntiring.",
        ),
    ),
}

EXTRA_SHARED: tuple[str, ...] = (
    "Barcha amaliyotlar mobil ilovada va {site} saytida mavjud.",
    "Murojaatlar kechayu kunduz qabul qilinadi, javob {hours} soat ichida beriladi.",
    "Murojaat holatini {order} raqami bo'yicha tekshirish mumkin.",
    "O'zgarishlar {date} sanasidan kuchga kiradi.",
    "Mijoz qaror haqida {email} manziliga xabar oladi.",
    "Bayram kunlari muddatlar uzayishi mumkin.",
    "Xavfsizlik uchun SMS kodlarni boshqalarga aytmang.",
    "Xizmat narxi tariflarda ko'rsatilgan va ogohlantirishsiz o'zgarmaydi.",
    "Hujjatlar elektron shaklda va bo'limda qabul qilinadi.",
    "Limitlar «{plan}» tarifi o'zgarganda qayta ko'riladi.",
    "Komissiya amaliyot o'tkazilgan paytda hisoblanadi.",
    "Amaliyotni faqat u o'tkazilgunga qadar bekor qilish mumkin.",
    "Ma'lumotlar shifrlangan holda saqlanadi va uchinchi shaxsga berilmaydi.",
)

EXTRA_CODE_SWITCH: tuple[tuple[str, str], ...] = (
    ("komissiya", "комиссия"),
    ("o'tkazma", "перевод"),
    ("karta", "карта"),
    ("muddat", "срок"),
    ("chegara", "лимит"),
    ("hisob", "счёт"),
    ("hujjat", "документ"),
    ("to'lov", "платёж"),
)


# Wave 3 (t22): loan/credit, deposits and remittance tracking archetypes.
from answerdesk.ingest.synthetic_data import wave3_uz as WAVE3
