interface MdastNode {
  type: string;
  value?: string;
  data?: {
    hChildren?: Array<{ value?: string }>;
  };
  children?: MdastNode[];
}

const BOLD_MARK = "**";
const BOLD_MATH_COMMAND = /^\\(?:boldsymbol|mathbf|bm)\s*\{/;

function walkParents(node: MdastNode, fn: (parent: MdastNode) => void) {
  if (!node.children) return;

  fn(node);
  for (const child of node.children) {
    walkParents(child, fn);
  }
}

function pushText(target: MdastNode[], value: string) {
  if (!value) return;

  const previous = target.at(-1);
  if (previous?.type === "text") {
    previous.value = `${previous.value ?? ""}${value}`;
    return;
  }

  target.push({ type: "text", value });
}

function fixLiteralStrong(parent: MdastNode) {
  const children = parent.children;
  if (!children?.some((child) => child.type === "text" && child.value?.includes(BOLD_MARK))) {
    return;
  }

  const nextChildren: MdastNode[] = [];
  let strongChildren: MdastNode[] | undefined;
  let changed = false;

  const currentTarget = () => strongChildren ?? nextChildren;

  for (const child of children) {
    if (child.type !== "text" || !child.value?.includes(BOLD_MARK)) {
      currentTarget().push(child);
      continue;
    }

    changed = true;
    const parts = child.value.split(BOLD_MARK);

    for (let i = 0; i < parts.length; i++) {
      pushText(currentTarget(), parts[i]);

      if (i === parts.length - 1) continue;

      if (strongChildren) {
        nextChildren.push({ type: "strong", children: strongChildren });
        strongChildren = undefined;
      } else {
        strongChildren = [];
      }
    }
  }

  if (strongChildren) {
    pushText(nextChildren, BOLD_MARK);
    nextChildren.push(...strongChildren);
  }

  if (changed) {
    parent.children = nextChildren;
  }
}

function fixStrongMath(node: MdastNode) {
  if (node.type !== "strong" || !node.children) return;

  for (const child of node.children) {
    if (child.type !== "inlineMath" || !child.value) continue;
    if (BOLD_MATH_COMMAND.test(child.value.trim())) continue;

    child.value = `\\boldsymbol{${child.value}}`;

    if (child.data?.hChildren?.[0]) {
      child.data.hChildren[0].value = child.value;
    }
  }
}

export function remarkFixCjkBold() {
  return function (tree: unknown) {
    walkParents(tree as MdastNode, (parent) => {
      fixLiteralStrong(parent);
      for (const child of parent.children ?? []) {
        fixStrongMath(child);
      }
    });
  };
}

export default remarkFixCjkBold;
