const CJK_CHARS_PER_MINUTE = 500;
const WORDS_PER_MINUTE = 220;

function normalizeMarkdown(source: string) {
  return source
    .replace(/```[\w-]*\r?\n/g, " ")
    .replace(/```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)]\([^)]*\)/g, "$1")
    .replace(/<[^>]*>/g, " ")
    .replace(/[$*_#>~|[\]{}()=+\-\\/.,!?;:"'`]/g, " ");
}

export function getReadingMinutes(source = "") {
  const text = normalizeMarkdown(source);
  const cjkChars =
    text.match(/[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff]/g)
      ?.length ?? 0;
  const latinText = text.replace(
    /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff]/g,
    " "
  );
  const words = latinText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g)?.length ?? 0;

  const minutes = cjkChars / CJK_CHARS_PER_MINUTE + words / WORDS_PER_MINUTE;
  return Math.max(1, Math.ceil(minutes));
}
