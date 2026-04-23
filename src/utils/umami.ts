/**
 * Umami Analytics API 유틸리티
 * 빌드 타임에 서버사이드로 Umami API를 호출하여 조회수/방문자 데이터를 가져옴
 * CORS 문제 없음 (서버에서 직접 호출)
 */

const UMAMI_API_URL = "https://api.umami.is/v1";

interface UmamiMetric {
  x: string;
  y: number;
}

interface UmamiStats {
  pageviews: number;
  visitors: number;
  visits: number;
  bounces: number;
  totaltime: number;
}

// 빌드 세션 내 캐시
let metricsCache: Record<string, number> | null = null;
let statsCache: UmamiStats | null = null;
let todayStatsCache: UmamiStats | null = null;

function getConfig() {
  const websiteId = import.meta.env.PUBLIC_UMAMI_WEBSITE_ID;
  const apiKey = import.meta.env.UMAMI_API_KEY;
  return { websiteId, apiKey };
}

/**
 * 전체 URL별 조회수 맵 가져오기
 * @returns { "/path": count, ... }
 */
export async function getPageViewMap(): Promise<Record<string, number>> {
  if (metricsCache) return metricsCache;

  const { websiteId, apiKey } = getConfig();
  if (!websiteId || !apiKey) return {};

  try {
    const now = Date.now();
    const res = await fetch(
      `${UMAMI_API_URL}/websites/${websiteId}/metrics?type=url&startAt=0&endAt=${now}&limit=500`,
      { headers: { "x-umami-api-key": apiKey } }
    );

    if (!res.ok) return {};

    const data: UmamiMetric[] = await res.json();
    const map: Record<string, number> = {};
    for (const item of data) {
      map[item.x] = item.y;
    }

    metricsCache = map;
    return map;
  } catch {
    return {};
  }
}

/**
 * 특정 페이지의 조회수 가져오기
 */
export async function getPageViewCount(path: string): Promise<number> {
  const map = await getPageViewMap();
  // trailing slash 유무 모두 확인
  return map[path] || map[path + "/"] || map[path.replace(/\/$/, "")] || 0;
}

/**
 * 전체 사이트 통계 가져오기
 */
export async function getSiteStats(): Promise<UmamiStats> {
  if (statsCache) return statsCache;

  const { websiteId, apiKey } = getConfig();
  const defaultStats: UmamiStats = {
    pageviews: 0,
    visitors: 0,
    visits: 0,
    bounces: 0,
    totaltime: 0,
  };

  if (!websiteId || !apiKey) return defaultStats;

  try {
    const now = Date.now();
    const res = await fetch(
      `${UMAMI_API_URL}/websites/${websiteId}/stats?startAt=0&endAt=${now}`,
      { headers: { "x-umami-api-key": apiKey } }
    );

    if (!res.ok) return defaultStats;

    statsCache = await res.json();
    return statsCache!;
  } catch {
    return defaultStats;
  }
}

/**
 * 오늘 방문자 통계 가져오기
 */
export async function getTodayStats(): Promise<UmamiStats> {
  if (todayStatsCache) return todayStatsCache;

  const { websiteId, apiKey } = getConfig();
  const defaultStats: UmamiStats = {
    pageviews: 0,
    visitors: 0,
    visits: 0,
    bounces: 0,
    totaltime: 0,
  };

  if (!websiteId || !apiKey) return defaultStats;

  try {
    const now = Date.now();
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);

    const res = await fetch(
      `${UMAMI_API_URL}/websites/${websiteId}/stats?startAt=${todayStart.getTime()}&endAt=${now}`,
      { headers: { "x-umami-api-key": apiKey } }
    );

    if (!res.ok) return defaultStats;

    todayStatsCache = await res.json();
    return todayStatsCache!;
  } catch {
    return defaultStats;
  }
}
