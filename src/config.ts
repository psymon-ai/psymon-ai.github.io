export const SITE = {
  website: "https://psymon-ai.github.io", // GitHub Pages URL
  author: "psymon",
  authorName: "박성열",
  profile: "https://github.com/psymon-ai",
  sameAs: ["https://github.com/psymon-ai", "https://huggingface.co/psymon"],
  desc: "로컬 AI와 LLM을 직접 만들고, 이해하는 과정을 기록한 psymon의 개발 블로그입니다.",
  title: "psymon-ai",
  ogImage: "About-me.png", // /public 폴더 기준 파일명
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
