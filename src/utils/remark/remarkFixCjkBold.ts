/**
 * remark-fix-cjk-bold
 *
 * CommonMark 스펙에서 닫는 ** 앞에 구두점(예: ')'), 뒤에 CJK 문자가 바로 오면
 * right-flanking delimiter로 인식하지 못해 볼드 처리가 안 되는 문제를 수정합니다.
 *
 * 파싱 후 AST의 텍스트 노드에 남아있는 **...** 패턴을 찾아
 * strong 노드로 변환합니다.
 */
import { visit } from "unist-util-visit";
import type { Root, Text, PhrasingContent } from "mdast";

// **...** 패턴 (텍스트 노드에 리터럴로 남아있는 것)
const UNPROCESSED_BOLD = /\*\*(.+?)\*\*/g;

export function remarkFixCjkBold() {
  return function (tree: Root) {
    visit(tree, "text", (node: Text, index, parent) => {
      if (!parent || index === undefined) return;

      const text = node.value;
      if (!text.includes("**")) return;

      const children: PhrasingContent[] = [];
      let lastIndex = 0;

      // Reset regex state
      UNPROCESSED_BOLD.lastIndex = 0;

      let match;
      while ((match = UNPROCESSED_BOLD.exec(text)) !== null) {
        // Add text before the match
        if (match.index > lastIndex) {
          children.push({
            type: "text",
            value: text.slice(lastIndex, match.index),
          });
        }

        // Add strong node
        children.push({
          type: "strong",
          children: [{ type: "text", value: match[1] }],
        });

        lastIndex = match.index + match[0].length;
      }

      if (children.length === 0) return; // No matches found

      // Add remaining text
      if (lastIndex < text.length) {
        children.push({
          type: "text",
          value: text.slice(lastIndex),
        });
      }

      // Replace the text node with the new children
      parent.children.splice(index, 1, ...children);

      // Return the index to continue visiting correctly
      return index + children.length;
    });
  };
}

export default remarkFixCjkBold;
