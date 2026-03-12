export const SITE = {
  website: "https://psymon-ai.github.io", // GitHub Pages URL
  author: "psymon",
  profile: "https://github.com/psymon-ai",
  desc: "A minimal, responsive and SEO-friendly Astro blog theme.",
  title: "psymon.ai",
  ogImage: "../public/About.png", // replace this with your og image path
  lightAndDarkMode: true,
  postPerIndex: 8,
  postPerPage: 8,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "ko", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Seoul", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
  comments: {
    enabled: true,
    provider: "giscus", // currently only giscus is supported
  },
} as const;
