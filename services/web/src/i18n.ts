/**
 * Nepali and English.
 *
 * Not an afterthought: the people operating this system are Nepali traffic
 * police and municipal staff, and an English-only console makes the software
 * harder to use for the people it is for. Nepali is the default.
 *
 * Deliberately a plain object rather than an i18n framework. The string count
 * is small, the two languages are known, and a dependency here would be more
 * code than it saves. Plate text itself is never translated — a plate is a
 * plate.
 */

export type Lang = "ne" | "en";

const STRINGS = {
  // Chrome
  "app.title": { en: "TheScanner", ne: "द स्क्यानर" },
  "app.subtitle": {
    en: "Vehicle movement intelligence",
    ne: "सवारी साधन आवागमन प्रणाली",
  },
  "nav.live": { en: "Live", ne: "प्रत्यक्ष" },
  "nav.alerts": { en: "Alerts", ne: "सतर्कता" },
  "nav.zones": { en: "Zones", ne: "क्षेत्र" },
  "nav.search": { en: "Search", ne: "खोज" },
  "nav.review": { en: "Review", ne: "समीक्षा" },
  "nav.anomalies": { en: "Anomalies", ne: "अनियमितता" },
  "nav.audit": { en: "Audit", ne: "लेखापरीक्षण" },

  // Auth
  "auth.username": { en: "Username", ne: "प्रयोगकर्ता नाम" },
  "auth.password": { en: "Password", ne: "पासवर्ड" },
  "auth.signIn": { en: "Sign in", ne: "साइन इन" },
  "auth.signOut": { en: "Sign out", ne: "साइन आउट" },
  "auth.failed": { en: "Sign-in failed", ne: "साइन इन असफल" },
  "auth.noMfa": {
    en: "Multi-factor authentication is not enrolled on this account.",
    ne: "यो खातामा बहु-कारक प्रमाणीकरण दर्ता गरिएको छैन।",
  },

  // Reads
  "read.plate": { en: "Plate", ne: "नम्बर प्लेट" },
  "read.camera": { en: "Camera", ne: "क्यामेरा" },
  "read.time": { en: "Time", ne: "समय" },
  "read.confidence": { en: "Confidence", ne: "विश्वसनीयता" },
  "read.frames": { en: "Frames", ne: "फ्रेम" },
  "read.owner": { en: "Ownership", ne: "स्वामित्व" },
  "read.none": { en: "No reads in this window", ne: "यस अवधिमा कुनै रेकर्ड छैन" },

  "conf.high": { en: "High", ne: "उच्च" },
  "conf.medium": { en: "Medium", ne: "मध्यम" },
  "conf.low": { en: "Low", ne: "न्यून" },
  "conf.reject": { en: "Rejected", ne: "अस्वीकृत" },

  "read.repaired": {
    en: "Inferred, not observed",
    ne: "अनुमानित, अवलोकन गरिएको होइन",
  },
  "read.repairedHelp": {
    en: "The plate grammar overrode what the pixels showed for these fields.",
    ne: "यी क्षेत्रहरूमा नम्बर प्लेटको ढाँचाले तस्बिरमा देखिएको कुरालाई प्रतिस्थापन गर्‍यो।",
  },
  "read.unverified": { en: "Signature unverified", ne: "हस्ताक्षर प्रमाणित छैन" },
  "read.unverifiedHelp": {
    en: "This read could not be verified against its camera's key. Do not act on it.",
    ne: "यो रेकर्ड क्यामेराको कुञ्जीसँग प्रमाणित हुन सकेन। यसमा कारबाही नगर्नुहोस्।",
  },

  // Search
  "search.plate": { en: "Plate number", ne: "नम्बर प्लेट" },
  "search.partial": { en: "Partial plate", ne: "आंशिक नम्बर प्लेट" },
  "search.reason": { en: "Reason for access", ne: "पहुँचको कारण" },
  "search.reasonHelp": {
    en: "Required and permanently logged against your name. Cite a case or authority reference.",
    ne: "अनिवार्य र तपाईंको नाममा स्थायी रूपमा अभिलेख गरिन्छ। मुद्दा वा अधिकार सन्दर्भ उल्लेख गर्नुहोस्।",
  },
  "search.go": { en: "Search", ne: "खोज्नुहोस्" },
  "search.results": { en: "results", ne: "परिणाम" },
  "search.convoy": { en: "Travelling with", ne: "सँगै यात्रा गर्ने" },

  // Zones
  "zone.occupancy": { en: "Current occupancy", ne: "हालको उपस्थिति" },
  "zone.entered": { en: "Entered", ne: "प्रवेश" },
  "zone.exited": { en: "Exited", ne: "निस्कियो" },
  "zone.dwell": { en: "Dwell", ne: "अवधि" },
  "zone.stillInside": { en: "Still inside", ne: "अझै भित्र" },
  "zone.trackLost": { en: "Track lost", ne: "ट्र्याक हरायो" },
  "zone.timedOut": { en: "Timed out", ne: "समय सकियो" },

  // Review
  "review.title": { en: "Human review queue", ne: "मानव समीक्षा सूची" },
  "review.help": {
    en: "Low-confidence reads. Corrections here become training data.",
    ne: "न्यून विश्वसनीयताका रेकर्ड। यहाँका सुधारहरू तालिम डेटा बन्छन्।",
  },
  "review.machineRead": { en: "Machine read", ne: "मेसिनले पढेको" },
  "review.confirm": { en: "Correct", ne: "सही" },
  "review.correct": { en: "Fix", ne: "सुधार" },
  "review.empty": { en: "Nothing awaiting review", ne: "समीक्षाको लागि केही छैन" },

  // Alerts
  "alert.actionable": { en: "Actionable", ne: "कारबाहीयोग्य" },
  "alert.reviewOnly": { en: "Review only", ne: "समीक्षा मात्र" },
  "alert.acknowledge": { en: "Acknowledge", ne: "स्वीकार" },
  "alert.empty": { en: "No outstanding alerts", ne: "कुनै बाँकी सतर्कता छैन" },

  // Anomalies
  "anom.colourClass": { en: "Colour disagrees with class letter", ne: "रङ र वर्ग अक्षर मेल खाँदैन" },
  "anom.plateVehicle": { en: "Plate class disagrees with vehicle", ne: "प्लेट वर्ग र सवारी मेल खाँदैन" },
  "anom.impossible": { en: "Physically impossible movement", ne: "भौतिक रूपमा असम्भव आवागमन" },
  "anom.registry": { en: "Registry mismatch", ne: "दर्ता विवरण मेल खाँदैन" },
  "anom.caution": {
    en: "Benign explanations exist. Review the evidence before acting.",
    ne: "सामान्य कारणहरू पनि हुन सक्छन्। कारबाही अघि प्रमाण जाँच्नुहोस्।",
  },

  // Stats
  "stats.reads": { en: "Reads (24h)", ne: "रेकर्ड (२४ घण्टा)" },
  "stats.plates": { en: "Distinct vehicles", ne: "फरक सवारी" },
  "stats.open": { en: "Currently in zones", ne: "क्षेत्रमा रहेका" },
  "stats.review": { en: "Awaiting review", ne: "समीक्षा बाँकी" },
  "stats.anomalies": { en: "Open anomalies", ne: "खुला अनियमितता" },
  "stats.unverified": { en: "Unverified reads", ne: "अप्रमाणित रेकर्ड" },

  // Generic
  "common.loading": { en: "Loading…", ne: "लोड हुँदै…" },
  "common.error": { en: "Something went wrong", ne: "केही गडबड भयो" },
  "common.retry": { en: "Retry", ne: "पुनः प्रयास" },
  "common.denied": { en: "You do not have permission for this", ne: "तपाईंलाई यसको अनुमति छैन" },
} as const;

export type StringKey = keyof typeof STRINGS;

export function t(key: StringKey, lang: Lang): string {
  return STRINGS[key][lang];
}

/** Devanagari digits, for rendering counts in Nepali.
 *  Plate serials are never converted — a plate is displayed exactly as read. */
const NE_DIGITS = "०१२३४५६७८९";

export function num(value: number, lang: Lang): string {
  const s = value.toLocaleString("en-US");
  if (lang !== "ne") return s;
  return s.replace(/[0-9]/g, (d) => NE_DIGITS[Number(d)]);
}

export function relativeTime(iso: string, lang: Lang): string {
  const then = new Date(iso).getTime();
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  const units: [number, string, string][] = [
    [60, "s", "से"],
    [3600, "m", "मि"],
    [86400, "h", "घ"],
  ];
  if (secs < 60) return lang === "ne" ? `${num(secs, lang)} से` : `${secs}s`;
  for (const [limit, en, ne] of units.slice(1)) {
    if (secs < limit) {
      const v = Math.round(secs / (limit / 60));
      return lang === "ne" ? `${num(v, lang)} ${ne}` : `${v}${en}`;
    }
  }
  const days = Math.round(secs / 86400);
  return lang === "ne" ? `${num(days, lang)} दिन` : `${days}d`;
}
