import { useEffect, useState } from "react";

/** Types out each word in sequence, pauses, deletes, moves to the next -- loops forever. */
export default function useTypewriter(words, { typeSpeed = 45, deleteSpeed = 25, pause = 1800 } = {}) {
  const [index, setIndex] = useState(0);
  const [text, setText] = useState("");
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const current = words[index % words.length];
    let timeout;

    if (!deleting && text === current) {
      timeout = setTimeout(() => setDeleting(true), pause);
    } else if (deleting && text === "") {
      setDeleting(false);
      setIndex((i) => (i + 1) % words.length);
    } else {
      const nextLength = deleting ? text.length - 1 : text.length + 1;
      timeout = setTimeout(() => {
        setText(current.slice(0, nextLength));
      }, deleting ? deleteSpeed : typeSpeed);
    }

    return () => clearTimeout(timeout);
  }, [text, deleting, index, words, typeSpeed, deleteSpeed, pause]);

  return text;
}
