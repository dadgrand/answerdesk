"""Turkish (Latin) content, wave 2: FAQ, fee table, app guide, fraud policy, closure, security.

LLM-assisted, not native-speaker validated — see ``docs/ingestion.md`` §12.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank

EXTRA_INTENTS: dict[str, IntentBank] = {
    "faq": IntentBank(
        title="Kartlar ve havaleler hakkında sık sorulan sorular",
        headings=("Genel sorular", "Ödeme ve limitler", "Belgeler ve süreler"),
        sentences=(
            "Ödeme geçmezse limiti ve kartın son kullanma tarihini kontrol edin.",
            "Telefon numarasına yapılan havale alıcının hesabına anında geçer.",
            "Hizmet bedeli {date} tarihinde alınır ve işlem sayısına bağlı değildir.",
            "Günlük havale limiti {limit} {currency}.",
            "Telefon numarasını değiştirmek için {doc} belgesi ve SMS onayı gerekir.",
            "Başvuru durumu {hours} saat içinde uygulamada güncellenir.",
            "Hesap hareketi dekontunu «Belgeler» bölümünden talep edebilirsiniz.",
            "İşlem {error} koduyla reddedilirse bilgileri kontrol edip yeniden gönderin.",
            "Hatalı çekilen tutar {days} iş günü içinde iade edilir.",
        ),
        faqs=(
            (
                "Ödeme neden reddedildi?",
                "Ödeme, {limit} {currency} limiti aşıldığında veya kart bloke olduğunda "
                "reddedilir.",
            ),
            (
                "Telefon numarasına havale ne kadar sürede ulaşır?",
                "Telefon numarasına yapılan havale alıcının hesabına anında geçer.",
            ),
        ),
        table_rows=(
            "Başvuru yanıtı|{days} gün|{channel}|{plan}",
            "Havale süresi|{hours} saat|{channel}|—",
        ),
        procedure_steps=(
            "«Destek» bölümünü açıp konuyu seçin.",
            "Durumu yazın ve {order} işlem numarasını ekleyin.",
        ),
    ),
    "fee_table": IntentBank(
        title="Ücretler ve komisyonlar: tam tablo",
        headings=("Ücret listesi", "Komisyon alınmayan durumlar", "Komisyon itirazı"),
        sentences=(
            "Başka bankaya havale komisyonu {fee} {currency}.",
            "Kendi bankamatiklerimizde nakit çekimde komisyon alınmaz.",
            "Kart hizmeti «{plan}» tarifesine dahildir ve ayrıca ücretlendirilmez.",
            "Acil havale için %{percent} komisyon, en az {fee} {currency}.",
            "Uluslararası havale komisyonu tutara ve para birimine bağlıdır.",
            "İnternette ödeme yaparken banka komisyon almaz.",
            "Ücret tarifeleri {days} günde birden fazla gözden geçirilmez.",
            "Hatalı alınan komisyon {days} gün içinde hesaba iade edilir.",
            "Ücretlerin tam listesi uygulamada ve {site} adresinde yer alır.",
        ),
        faqs=(
            (
                "Başka bankaya havale ne kadar?",
                "Başka bankaya havale komisyonu {fee} {currency}.",
            ),
            (
                "Alınan komisyon iade edilebilir mi?",
                "Hatalı alınan komisyon {days} gün içinde hesaba iade edilir.",
            ),
        ),
        table_rows=(
            "Başka bankaya havale|{fee} {currency}|{days} gün|{channel}",
            "Acil havale|{percent}%|anında|{channel}",
        ),
        procedure_steps=(
            "Hesap hareketlerinde komisyon alınan işlemi bulun.",
            "{order} işlem numarasıyla başvuru oluşturun.",
        ),
    ),
    "app_guide": IntentBank(
        title="Mobil uygulama: adım adım kılavuz",
        headings=("Giriş ve ayarlar", "Ödemeler ve havaleler", "Uygulamada güvenlik"),
        sentences=(
            "Uygulamaya giriş SMS kodu veya biyometri ile onaylanır.",
            "Alıcı eklemek için «Havale» bölümünü açın ve «Yeni alıcı»ya dokunun.",
            "İşlem geçmişi tarih, tutar ve ödeme türüne göre filtrelenebilir.",
            "Harcama bildirimleri profil ayarlarından açılır.",
            "Ödeme şablonu ilk ödemeden sonra otomatik kaydedilir.",
            "Uygulamanın güncellenmesi zayıf bağlantıda {hours} saat sürebilir.",
            "Cihaz değiştiğinde {phone} numarasını yeniden onaylayın.",
            "Uygulama açılmıyorsa sürümü ve boş alanı kontrol edin.",
            "Destek sohbeti uygulamada günün her saatinde açıktır.",
        ),
        faqs=(
            (
                "Havale için alıcıyı nasıl eklerim?",
                "«Havale» bölümünü açın, «Yeni alıcı»ya dokunup bilgileri girin.",
            ),
            (
                "Uygulama açılmıyorsa ne yapmalıyım?",
                "Sürümü ve boş alanı kontrol edip yeniden giriş yapın.",
            ),
        ),
        table_rows=(
            "Biyometri ile giriş|mevcut|{plan}|{channel}",
            "Uygulama güncellemesi|{hours} saat|{channel}|—",
        ),
        procedure_steps=(
            "Uygulamayı açıp profilinize giriş yapın.",
            "«Havale» bölümünü seçip formu doldurun.",
        ),
    ),
    "fraud_policy": IntentBank(
        title="Dolandırıcılıkla mücadele politikası",
        headings=("Dolandırıcılık belirtileri", "Banka ne yapar", "Müşteri ne yapmalı"),
        sentences=(
            "Banka telefonda hiçbir zaman şifre veya SMS kodu istemez.",
            "Şüpheli işlemler {hours} saate kadar otomatik olarak bloke edilir.",
            "Müşteri SMS kodunu paylaşırsa kart derhal bloke edilmelidir.",
            "Para iadesi başvurusu {days} iş günü içinde incelenir.",
            "Arayan numara kaydedilir ve güvenlik birimine iletilir.",
            "Dolandırıcıya gönderilen tutara {days} gün içinde itiraz edilebilir.",
            "Uygulamaya girişlerde {email} adresine bildirim gönderilir.",
            "Çalışanı yalnızca {site} adresindeki {phone} numarasından doğrulayın.",
            "Politika yeni dolandırıcılık yöntemleri dikkate alınarak yılda bir güncellenir.",
        ),
        faqs=(
            (
                "Banka SMS kodu isteyebilir mi?",
                "Hayır, banka telefonda hiçbir zaman şifre veya SMS kodu istemez.",
            ),
            (
                "Kodu dolandırıcıya verdim, ne yapmalıyım?",
                "Kartı hemen uygulamadan bloke edin ve {phone} numarasını arayın.",
            ),
        ),
        table_rows=(
            "Şüpheli işlem bloke süresi|{hours} saat|{channel}|{plan}",
            "Para iadesi|{days} gün|{channel}|—",
        ),
        procedure_steps=(
            "Kartı uygulamadan bloke edin.",
            "{phone} numarasını arayıp {order} işlem kodunu bildirin.",
        ),
    ),
    "account_closure": IntentBank(
        title="Hesap kapatma ve sözleşme feshi",
        headings=("Hesap ne zaman kapatılır", "Kapatma süreci", "Sonrasında ne olur"),
        sentences=(
            "Hesap, tüm işlemler tamamlandıktan sonra kapatılabilir.",
            "Kapatma başvurusu şubede veya uygulamada yapılır.",
            "Hesap kapatma süresi {days} iş gününe kadardır.",
            "Kalan bakiye bildirilen hesaba aktarılır veya nakit ödenir.",
            "Hizmet borcu kapatma sırasında bakiyeden mahsup edilir.",
            "Hesap kapatıldığında sözleşme feshedilir ve kart bloke edilir.",
            "Kapatma belgesi {days} gün içinde düzenlenir.",
            "Hesap aynı gün yeniden açılabilir.",
            "Hesabın kapatılması önceki kredi yükümlülüklerini ortadan kaldırmaz.",
        ),
        faqs=(
            (
                "Hesap kapatma ne kadar sürer?",
                "Başvurudan sonra hesap {days} iş günü içinde kapatılır.",
            ),
            (
                "Kapatmada kalan bakiye nereye aktarılır?",
                "Kalan bakiye bildirilen hesaba aktarılır veya nakit ödenir.",
            ),
        ),
        table_rows=(
            "Hesap kapatma|{days} gün|{channel}|{plan}",
            "Kapatma belgesi|{days} gün|{channel}|—",
        ),
        procedure_steps=(
            "Hesaptaki tüm işlemlerin tamamlandığını kontrol edin.",
            "Kapatma başvurusunu verin ve bakiye için hesap bilgisi paylaşın.",
        ),
    ),
    "security_notice": IntentBank(
        title="Güvenlik: kodlar, şifreler ve şüpheli aramalar",
        headings=("Güvenlik kuralları", "Kodlar ve şifreler", "Nereye bildirmeli"),
        sentences=(
            "SMS kodunu kimseye vermeyin, arayan kendini banka çalışanı olarak tanıtsa bile.",
            "Uygulama şifresi yalnızca müşteride saklanır ve destek ekibi istemez.",
            "Yeni cihazdan giriş ek onay gerektirir.",
            "Şüpheli bir arama gelirse görüşmeyi bitirip {phone} numarasını arayın.",
            "Çalışanı {site} adresindeki {phone} numarasından doğrulayabilirsiniz.",
            "Şüpheli işlemleri {email} adresine bildirin.",
            "SMS kodu {hours} saat geçerlidir ve bir kez kullanılır.",
            "Üç hatalı girişten sonra uygulama {hours} saat kilitlenir.",
            "Güvenlik bilgileri {site} adresinin «Yardım» bölümünde yayımlanır.",
        ),
        faqs=(
            (
                "Banka çalışanı SMS kodu isterse bu normal mi?",
                "Hayır, SMS kodunu banka çalışanı dahil kimseye vermeyin.",
            ),
            (
                "Şüpheli aramayı nereye bildiririm?",
                "Görüşmeyi bitirip {site} adresindeki {phone} numarasını arayın.",
            ),
        ),
        table_rows=(
            "Kod geçerlilik süresi|{hours} saat|{channel}|{plan}",
            "Hatalı giriş sonrası kilit|{hours} saat|{channel}|—",
        ),
        procedure_steps=(
            "Şüpheli görüşmeyi sonlandırın.",
            "{phone} numarasını arayıp durumu anlatın.",
        ),
    ),
}

EXTRA_SHARED: tuple[str, ...] = (
    "Tüm işlemler mobil uygulamada ve {site} adresinde yapılabilir.",
    "Başvurular günün her saatinde alınır, yanıt {hours} saat içinde verilir.",
    "Başvuru durumunu {order} numarasıyla kontrol edebilirsiniz.",
    "Değişiklikler {date} tarihinde yürürlüğe girer.",
    "Müşteriye karar {email} adresinden bildirilir.",
    "Tatil günlerinde süreler uzayabilir.",
    "Güvenlik için SMS kodlarını başkalarıyla paylaşmayın.",
    "Hizmet bedeli tarifelerde belirtilir ve bildirim olmadan değişmez.",
    "Belgeler elektronik olarak ve şubede kabul edilir.",
    "Limitler «{plan}» tarifesi değiştiğinde yeniden değerlendirilir.",
    "Komisyon işlem gerçekleştiği anda hesaplanır.",
    "İşlem yalnızca gerçekleşmeden önce iptal edilebilir.",
    "Veriler şifreli saklanır ve üçüncü taraflarla paylaşılmaz.",
)

EXTRA_CODE_SWITCH: tuple[tuple[str, str], ...] = (
    ("komisyon", "комиссия"),
    ("havale", "перевод"),
    ("kart", "карта"),
    ("süre", "срок"),
    ("limit", "лимит"),
    ("hesap", "счёт"),
    ("belge", "документ"),
    ("ödeme", "платёж"),
)


# Wave 3 (t22): loan/credit, deposits and remittance tracking archetypes.
from answerdesk.ingest.synthetic_data import wave3_tr as WAVE3
