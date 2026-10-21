"""Turkish (Latin) content, wave 3: loan/credit, deposits, remittance tracking (t22).

LLM-assisted, not native-speaker validated — see ``docs/ingestion.md`` §12.
"""

from __future__ import annotations

from answerdesk.ingest.synthetic import IntentBank

EXTRA_INTENTS: dict[str, IntentBank] = {
    "loan_credit": IntentBank(
        title="Nakit kredi: başvuru, faiz ve erken ödeme",
        headings=("Başvuru nasıl yapılır", "Faiz ve onay koşulları", "Erken ödeme"),
        sentences=(
            "Kredi başvurusu {days} iş günü içinde değerlendirilir.",
            "Kredi faizi %{percent}, tutar {limit} {currency} tutarına kadar.",
            "Onay için {doc} belgesi ve gelir teyidi gerekir.",
            "Karar {hours} saat içinde uygulamada ve {email} adresinde görünür.",
            "Kredi her gün komisyonsuz olarak erken kapatılabilir.",
            "Erken ödemede faiz gerçek süre üzerinden yeniden hesaplanır.",
            "Ödeme planı uygulamada ve şubede görüntülenebilir.",
        ),
        faqs=(
            (
                "Kredi başvurusu nasıl yapılır?",
                "Başvuru uygulamadan veya şubeden yapılır ve {days} iş günü içinde değerlendirilir.",
            ),
            (
                "Kredi erken ödenebilir mi?",
                "Evet, kredi her gün komisyonsuz olarak erken kapatılabilir.",
            ),
        ),
        table_rows=(
            "Kredi faizi|{percent}%|{days} gün|{plan}",
            "Kredi tutarı|{limit} {currency}|{days} güne kadar|—",
        ),
        procedure_steps=(
            "Başvuruyu doldurun ve {doc} belgesini ekleyin.",
            "{order} başvuru numarasıyla kararı bekleyin.",
        ),
    ),
    "deposit_savings": IntentBank(
        title="Mevduat ve birikim hesapları",
        headings=("Mevduat açma", "Faiz ve para ekleme", "Çekme ve kapatma"),
        sentences=(
            "Mevduat uygulamadan veya şubeden {days} günden başlayarak açılır.",
            "Mevduat faizi %{percent}.",
            "Asgari tutar — {amount} {currency}.",
            "Mevduat ve depozitoyu online açabilirsiniz, para aynı gün geçer.",
            "İlk {days} gün içinde para ekleme sınırsızdır.",
            "Faiz her ay hesaplanır ve kapitalize edilir.",
            "Vadeden önce kısmi çekim faizi %{percent} düzeyine indirir.",
            "Mevduatın kapatılması {hours} saat sürer.",
        ),
        faqs=(
            (
                "Mevduat nasıl açılır?",
                "Mevduat uygulamadan veya şubeden {days} günden başlayarak açılır.",
            ),
            (
                "Vadeden önce çekersem ne olur?",
                "Vadeden önce kısmi çekim faizi %{percent} düzeyine indirir.",
            ),
        ),
        table_rows=(
            "Mevduat faizi|{percent}%|{days} gün|{plan}",
            "Asgari tutar|{amount} {currency}|{hours} saat|—",
        ),
        procedure_steps=(
            "Vadeyi ve tutarı seçin.",
            "Açılışı SMS kodu ile onaylayın.",
        ),
    ),
    "remittance_tracking": IntentBank(
        title="Havale takibi: durum, araştırma, iade",
        headings=("Havale durumu", "Para ulaşmadı", "Araştırma ve iade"),
        sentences=(
            "Havale durumu {order} işlem numarasıyla kontrol edilir.",
            "Başka bankaya havale {hours} saat içinde ulaşır.",
            "Tacikistan'a havale genellikle {hours} saat içinde ulaşır.",
            "Para yakınınıza ulaşmadıysa havale durumunu kontrol edin.",
            "Para {days} gün içinde ulaşmazsa araştırma başvurusu yapılır.",
            "Araştırma {days} ile {count} iş günü arasında sürer.",
            "Para, alıcı banka onayladıktan sonra iade edilir.",
            "Hata bankadan kaynaklanıyorsa araştırma için komisyon alınmaz.",
            "Araştırma durumu uygulamada ve {phone} numarasından güncellenir.",
        ),
        faqs=(
            (
                "Para ulaşmadı, ne yapmalıyım?",
                "Havale durumunu {order} numarasıyla kontrol edin; {days} gün içinde ulaşmazsa "
                "araştırma başvurusu yapın.",
            ),
            (
                "Havale araştırması ne kadar sürer?",
                "Araştırma {days} ile {count} iş günü arasında sürer.",
            ),
        ),
        table_rows=(
            "Havale ulaşması|{hours} saat|{days} gün|{channel}",
            "Araştırma ve iade|{days} gün|{count} gün|{channel}",
        ),
        procedure_steps=(
            "{order} numarasıyla işlemi açın.",
            "Para ulaşmadıysa araştırma başvurusu yapın.",
        ),
    ),
}
