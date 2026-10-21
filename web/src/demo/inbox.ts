/**
 * The operator inbox (demo data).
 *
 * The answerdesk service owns per-turn endpoints but has no queue API in v1.0: a real
 * deployment's front end (helpdesk, CRM) pushes incoming turns into the console. The demo
 * therefore ships a synthetic inbox — six locales, one PII guard case, mixed SLA states —
 * and every message in it is *sent to the real service* when the operator opens the
 * conversation (or answered by MSW in mock mode). Nothing here is a canned answer.
 *
 * All names and identifiers are fictional (requirements C-10, §13.4); the card number is
 * the ISO test range, not a real account.
 */

import type { IntentBucket, Priority } from "../api/types";

export interface InboxMessage {
  role: "customer" | "agent";
  text: string;
  at: number;
}

export interface Conversation {
  id: string;
  customer: {
    name: string;
    initials: string;
    locale: string;
    plan: string;
    tenure: string;
  };
  channel: string;
  subject: string;
  messages: InboxMessage[];
  /** Epoch ms when the current customer message started waiting (drives the SLA timer). */
  waitingSince: number;
  priority: Priority;
  intentBucket: IntentBucket;
  intentLabel: string | null;
  /** The message that is sent to the answer pipeline when the conversation is opened. */
  latestMessage: string;
}

const SECOND = 1000;

export function buildInbox(now: number = Date.now()): Conversation[] {
  return [
    {
      id: "c-1042",
      customer: { name: "Aisha Rahimova", initials: "AR", locale: "ru", plan: "Premium", tenure: "2y 4m" },
      channel: "Web chat",
      subject: "Card servicing fee",
      messages: [
        { role: "customer", text: "Здравствуйте! Подскажите, пожалуйста, по дебетовой карте «Вектор».", at: now - 98 * SECOND },
        { role: "customer", text: "обслуживание карты", at: now - 42 * SECOND },
      ],
      waitingSince: now - 42 * SECOND,
      priority: "normal",
      intentBucket: "informational",
      intentLabel: "pricing",
      latestMessage: "обслуживание карты",
    },
    {
      id: "c-1043",
      customer: { name: "Mehrzod Sattorov", initials: "MS", locale: "tg", plan: "Standard", tenure: "8m" },
      channel: "Telegram",
      subject: "Replacing a card",
      messages: [
        { role: "customer", text: "Салом! Кортҳоро чӣ тавр иваз кардан мумкин аст?", at: now - 88 * SECOND },
      ],
      waitingSince: now - 88 * SECOND,
      priority: "high",
      intentBucket: "transactional",
      intentLabel: "card_replacement",
      latestMessage: "Кортҳоро чӣ тавр иваз кардан мумкин аст?",
    },
    {
      id: "c-1044",
      customer: { name: "Dilnoza Karimova", initials: "DK", locale: "uz", plan: "Premium", tenure: "1y 1m" },
      channel: "Mobile app",
      subject: "Refund timing",
      messages: [
        { role: "customer", text: "To‘lovni qaytarish qancha vaqt oladi?", at: now - 148 * SECOND },
      ],
      waitingSince: now - 148 * SECOND,
      priority: "urgent",
      intentBucket: "transactional",
      intentLabel: "refund",
      latestMessage: "To‘lovni qaytarish qancha vaqt oladi?",
    },
    {
      id: "c-1045",
      customer: { name: "Aibek Osmonov", initials: "AO", locale: "ky", plan: "Standard", tenure: "3m" },
      channel: "Web chat",
      subject: "Card replacement",
      messages: [
        { role: "customer", text: "Кортону кантип алмаштырса болот?", at: now - 61 * SECOND },
      ],
      waitingSince: now - 61 * SECOND,
      priority: "normal",
      intentBucket: "informational",
      intentLabel: "card_replacement",
      latestMessage: "Кортону кантип алмаштырса болот?",
    },
    {
      id: "c-1046",
      customer: { name: "Elif Yilmaz", initials: "EY", locale: "tr", plan: "Premium", tenure: "5y" },
      channel: "Telegram",
      subject: "Card limit",
      messages: [
        { role: "customer", text: "Kartımın limitini nasıl artırabilirim?", at: now - 33 * SECOND },
      ],
      waitingSince: now - 33 * SECOND,
      priority: "normal",
      intentBucket: "informational",
      intentLabel: "limits",
      latestMessage: "Kartımın limitini nasıl artırabilirim?",
    },
    {
      id: "c-1047",
      customer: { name: "Li Wei", initials: "LW", locale: "zh", plan: "Standard", tenure: "6m" },
      channel: "Web chat",
      subject: "退款时长",
      messages: [{ role: "customer", text: "退款需要多长时间？", at: now - 205 * SECOND }],
      waitingSince: now - 205 * SECOND,
      priority: "urgent",
      intentBucket: "transactional",
      intentLabel: "refund",
      latestMessage: "退款需要多长时间？",
    },
    {
      id: "c-1048",
      customer: { name: "Igor Petrov", initials: "IP", locale: "ru", plan: "Standard", tenure: "11m" },
      channel: "Web chat",
      subject: "Balance check",
      messages: [
        {
          role: "customer",
          text: "Карта 4111 1111 1111 1111, проверьте баланс пожалуйста",
          at: now - 12 * SECOND,
        },
      ],
      waitingSince: now - 12 * SECOND,
      priority: "high",
      intentBucket: "account_legal",
      intentLabel: "account",
      latestMessage: "Карта 4111 1111 1111 1111, проверьте баланс пожалуйста",
    },
  ];
}

export function conversationById(inbox: Conversation[], id: string | null): Conversation | null {
  if (!id) return null;
  return inbox.find((conversation) => conversation.id === id) ?? null;
}
