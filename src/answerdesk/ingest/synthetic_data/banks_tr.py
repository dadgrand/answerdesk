"""Turkish (tr) content bank for the synthetic support corpus (FR-ING-10).

Fictional products, reserved identifiers only (§13.4); no real institution or brand is
named. Every ``{slot}`` is provided by the generator.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank, LocaleBank

_INTENTS: dict[str, IntentBank] = {
    "tariffs": IntentBank(
        title="Tarifeler ve hizmet ücretleri",
        headings=("Tarifelere genel bakış", "Aylık ücret ve komisyon", "Tarife değişikliği"),
        sentences=(
            "«{plan}» tarifesi ayda {amount} {currency} tutarındadır ve {count} ücretsiz işlem "
            "içerir.",
            "«{plan}» tarifesine geçişte ilk ay ücretsizdir.",
            "Aylık ücret, hesapta yeterli bakiye varsa {date} tarihinde alınır.",
            "Tarife {days} günde bir defadan fazla değiştirilemez.",
            "«{plan}» tarifesine kart hizmeti ve mobil uygulama erişimi dahildir.",
            "Limit üstü işlem komisyonu {fee} {currency} tutarındadır.",
            "Günlük nakit çekim limiti {limit} {currency}.",
            "Yıllık ödemede %{percent} indirim uygulanır.",
        ),
        faqs=(
            (
                "«{plan}» tarifesinin hizmet ücreti ne kadar?",
                "«{plan}» tarifesinin hizmet ücreti ayda {amount} {currency} tutarındadır.",
            ),
            (
                "Tarife uygulamadan değiştirilebilir mi?",
                "Evet, tarife uygulamadan değiştirilebilir ancak {days} günde bir defadan fazla "
                "değil.",
            ),
        ),
        table_rows=(
            "Aylık ücret|{amount} {currency}|{date}|{plan}",
            "Limit üstü komisyon|{fee} {currency}|{days} gün|{channel}",
        ),
        procedure_steps=(
            "Uygulamada «Tarifeler» bölümünü açın.",
            "«{plan}» tarifesini seçip «Bağlan» düğmesine dokunun.",
            "Değişikliği SMS kodu ile onaylayın.",
        ),
    ),
    "cards": IntentBank(
        title="Kartlar: başvuru, limitler ve blokaj",
        headings=("Kart başvurusu", "Kart limitleri", "Blokaj ve yeniden basım"),
        sentences=(
            "Kart, başvuru onaylandıktan sonra {days} iş günü içinde basılır.",
            "Kartın günlük limiti {limit} {currency}, gece limiti {amount} {currency}.",
            "Diğer bankamatiklerde nakit çekim komisyonu {fee} {currency}.",
            "Kart kaybolursa uygulamadan bloke edin ve yeniden basım talep edin.",
            "Yeniden basım {amount} {currency} tutarındadır.",
            "{card} numaralı kart ile internet ödemesi yapılabilir.",
            "Şüpheli işlemde kart {hours} saat boyunca otomatik olarak bloke edilir.",
            "PIN kodu bankamatikte veya uygulamada değiştirilebilir.",
        ),
        faqs=(
            (
                "Kartın yeniden basımı ne kadar?",
                "Yeniden basım {amount} {currency} tutarındadır ve {days} gün sürer.",
            ),
            (
                "Kart kaybolursa ne yapmalıyım?",
                "Kartı uygulamadan bloke edin ve yeniden basım talep edin; işlem {days} gün sürer.",
            ),
        ),
        table_rows=(
            "Günlük limit|{limit} {currency}|kart|{plan}",
            "Yeniden basım|{amount} {currency}|{days} gün|{channel}",
        ),
        procedure_steps=(
            "Kartı uygulamadan bloke edin.",
            "Yeniden basım talep edin ve teslimat adresini girin.",
            "Kartı {days} iş günü içinde bekleyin.",
        ),
    ),
    "transfers": IntentBank(
        title="Transferler: süre, komisyon ve limitler",
        headings=("Yurt içi transferler", "Uluslararası transferler", "Durumlar ve hatalar"),
        sentences=(
            "Telefon numarasına transfer anında ve komisyonsuz gerçekleşir.",
            "Uluslararası transfer {days} ile {count} iş günü arasında sürer.",
            "Uluslararası transfer komisyonu tutarın %{percent} kadarı, en az {fee} {currency}.",
            "Tek transferin azami tutarı {limit} {currency}.",
            "Transfer ulaşmadıysa {order} numarası ile durumu kontrol edin.",
            "{error} hatası alıcının bulunamadığını gösterir.",
            "Transfer yalnızca ödeme sistemine gönderilmeden önce iptal edilebilir.",
            "Günlük transfer limiti {limit} {currency}; geçmiş {count} ay için görüntülenebilir.",
        ),
        faqs=(
            (
                "Uluslararası transfer ne kadar sürer?",
                "Uluslararası transfer {days} ile {count} iş günü arasında sürer.",
            ),
            (
                "{error} hatası ne anlama gelir?",
                "{error} hatası alıcının bulunamadığını veya bilgilerin hatalı olduğunu gösterir.",
            ),
        ),
        table_rows=(
            "Yurt içi transfer|anında|{fee} {currency}|{channel}",
            "Uluslararası transfer|{days} gün|{percent}%|{channel}",
        ),
        procedure_steps=(
            "Alıcı bilgilerini kontrol edin.",
            "Tutarı ve komisyonu onaylayın.",
            "Takip için {order} işlem numarasını saklayın.",
        ),
    ),
    "documents": IntentBank(
        title="Belgeler: dekont, ekstre ve yenileme",
        headings=("Dekont ve ekstre", "Elektronik belgeler", "Süre ve ücret"),
        sentences=(
            "Hesap hareketi dekontu {days} iş günü içinde hazırlanır.",
            "Elektronik ekstre uygulamada son {count} ay için görüntülenebilir.",
            "Kâğıt dekont ücreti {amount} {currency}.",
            "{doc} belgesi yalnızca kimlik doğrulamasından sonra talep edilebilir.",
            "Hazır belge şubeden iş günlerinde saat {time} kadar alınabilir.",
            "Belgenin elektronik imzası {site} adresinde doğrulanır.",
            "Kimlik bilgileri değişirse belgeler {days} gün içinde güncellenmelidir.",
            "Belge kopyası «Belgeler» bölümünde {date} tarihine kadar görüntülenebilir.",
        ),
        faqs=(
            (
                "Hesap hareketi dekontunu nasıl alırım?",
                "Dekont {days} iş günü içinde hazırlanır; uygulamadan veya şubeden talep "
                "edebilirsiniz.",
            ),
            (
                "Kâğıt ekstre ücreti ne kadar?",
                "Kâğıt ekstre {amount} {currency}, elektronik ekstre ücretsizdir.",
            ),
        ),
        table_rows=(
            "Elektronik dekont|ücretsiz|{days} gün|{channel}",
            "Kâğıt dekont|{amount} {currency}|{days} gün|şube",
        ),
        procedure_steps=(
            "Uygulamada «Belgeler» bölümünü açın.",
            "{doc} belgesini ve teslim yöntemini seçin.",
            "Talep durumunu {order} numarası ile kontrol edin.",
        ),
    ),
    "kyc": IntentBank(
        title="Kimlik doğrulama ve bilgi teyidi",
        headings=("Hangi bilgiler gerekli", "Doğrulama süresi", "Ret ve yeniden başvuru"),
        sentences=(
            "Kimlik doğrulama {hours} saat, nadiren {days} gün sürer.",
            "{doc} fotoğrafı, selfie ve adres teyidi gereklidir.",
            "Adres, {count} aydan eski olmayan bir belge ile teyit edilir.",
            "Bilgiler uyuşmazsa doğrulama {error} kodu ile reddedilir.",
            "Yeniden başvuru, ret işleminden {hours} saat sonra yapılabilir.",
            "Bilgiler şifreli saklanır ve onay olmadan üçüncü taraflara verilmez.",
            "Belge fotoğrafı renkli ve parlamasız olmalıdır.",
            "Ret itirazı {days} iş günü içinde incelenir.",
        ),
        faqs=(
            (
                "Kimlik doğrulama ne kadar sürer?",
                "Doğrulama {hours} saat, bazen {days} gün sürer.",
            ),
            (
                "Ret durumunda ne yapmalıyım?",
                "Bilgileri kontrol edip {hours} saat sonra {error} kodu ile yeniden başvurun.",
            ),
        ),
        table_rows=(
            "Kimlik doğrulama|{hours} saat|{channel}|{doc}",
            "Yeniden başvuru|{hours} saat|{channel}|—",
        ),
        procedure_steps=(
            "{doc} fotoğrafını ve selfie hazırlayın.",
            "Adres ve iletişim bilgilerini doldurun.",
            "Başvuruyu gönderin ve sonucu bekleyin.",
        ),
    ),
    "complaints": IntentBank(
        title="Başvurular ve şikâyetler: süreç ve süreler",
        headings=("Başvuru nasıl yapılır", "İnceleme süresi", "Eskalasyon ve kararlar"),
        sentences=(
            "Başvuru uygulamadan, {site} adresinden veya {phone} numarasından yapılabilir.",
            "Başvurunun incelenme süresi {days} iş gününe kadardır.",
            "{error} hatasına ilişkin acil başvurular {hours} saat içinde incelenir.",
            "Başvuruya {order} işlem numarasını ve sorun açıklamasını ekleyin.",
            "Başvuru durumu «Destek» bölümünde güncellenir.",
            "Karar {email} adresine ve uygulamaya gönderilir.",
            "Karar uygun değilse kıdemli operatöre başvurun.",
            "Aynı konudaki tekrarlanan başvuru, önceki karara atıfla kapatılır.",
        ),
        faqs=(
            (
                "Komisyon şikâyetini nasıl yaparım?",
                "Başvuruyu uygulamadan veya {site} adresinden yapın ve {order} işlem numarasını "
                "ekleyin.",
            ),
            (
                "Başvuru ne kadar sürede incelenir?",
                "Genellikle {days} iş gününe kadar; acil başvurular {hours} saat içinde.",
            ),
        ),
        table_rows=(
            "Normal başvuru|{days} gün|{channel}|—",
            "Acil başvuru|{hours} saat|{channel}|{error}",
        ),
        procedure_steps=(
            "Sorunu açıklayın ve {order} işlem numarasını ekleyin.",
            "İletişim kanalını seçin: {channel}.",
            "Başvuru durumunu uygulamadan izleyin.",
        ),
    ),
}

BANK: LocaleBank = LocaleBank(
    code="tr",
    display_name="Turkish",
    currency="TRY",
    contact_heading="İletişim ve çalışma saatleri",
    contact_lines=(
        "Destek telefonu: {phone}.",
        "E-posta: {email}.",
        "Web sitesi: {site}. Her gün {time} ile 20:00 arasında hizmet veriyoruz.",
        "{date} itibarıyla geçerlidir.",
        "Başvuru numarası: {order}.",
        "Yaygın hata kodu: {error}.",
    ),
    faq_heading="Sık sorulan sorular",
    table_heading="Tarifeler ve limitler",
    generic_headings=(
        "Genel koşullar",
        "Nasıl çalışır",
        "Süre ve durum",
        "Güvenlik",
        "Sonraki adım",
    ),
    generic_sentences=(
        "Tüm işlemler mobil uygulamada ve {site} adresinde yapılabilir.",
        "Bir sorun olursa {phone} numarasından desteğe ulaşın.",
        "Başvuruları iş günlerinde {time} ile 20:00 arasında inceliyoruz.",
        "Durumu {order} numarası ile kontrol edebilirsiniz.",
        "Değişiklikler {date} tarihinde yürürlüğe girer.",
        "Müşteriye karar {email} adresinden bildirilir.",
    ),
    procedure_heading="İşlem adımları",
    procedure_steps=(
        "Uygulamayı açın ve hesabınıza giriş yapın.",
        "İlgili bölümü seçin ve formu doldurun.",
        "İşlemi SMS kodu ile onaylayın.",
        "Başvuru numarasını saklayın: {order}.",
    ),
    table_header=("Gösterge", "Değer", "Süre", "Not"),
    table_rows=(
        "Komisyon|{fee} {currency}|{days} gün|{channel}",
        "Limit|{limit} {currency}|günlük|{plan}",
        "İşlem süresi|{hours} saat|—|{channel}",
        "Dekont ücreti|{amount} {currency}|{days} gün|{doc}",
        "Komisyon oranı|{percent}%|—|{channel}",
    ),
    plan_names=("Başlangıç", "Temel", "Konfor", "Premium"),
    channels=("uygulama", "web sitesi", "telefon", "şube"),
    document_names=("pasaport", "kimlik kartı", "sürücü belgesi"),
    code_switch_pairs=(
        ("komisyon", "fee"),
        ("havale", "transfer"),
        ("kart", "card"),
        ("sınır", "limit"),
        ("süre", "deadline"),
        ("tarife", "plan"),
    ),
    date_templates=("{date} itibarıyla.", "{date} tarihine kadar.", "(tarih: {date})"),
    intents=_INTENTS,
)
