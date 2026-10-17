  (function () {
    const parseValue = (text) => {
      const raw = text.trim();
      if (!raw || raw === "-") return null;

      const minuteMatch = raw.match(/^([0-9]*\.?[0-9]+)\s*min$/i);
      if (minuteMatch) return parseFloat(minuteMatch[1]) * 60;

      const secondMatch = raw.match(/^([0-9]*\.?[0-9]+)\s*s$/i);
      if (secondMatch) return parseFloat(secondMatch[1]);

      // Rank multi-seed results by their mean while retaining the uncertainty.
      const meanMatch = raw.match(/^(-?[0-9]*\.?[0-9]+)\s*±\s*[0-9]*\.?[0-9]+$/);
      if (meanMatch) return parseFloat(meanMatch[1]);

      const numericMatch = raw.match(/^-?[0-9]*\.?[0-9]+$/);
      if (numericMatch) return parseFloat(raw);

      return null;
    };

    const uniqueSorted = (values, lowerIsBetter) => {
      const uniq = [...new Set(values)];
      uniq.sort((a, b) => lowerIsBetter ? a - b : b - a);
      return uniq;
    };

    const applyScoreMark = (cell, kind) => {
      const text = cell.textContent.trim();
      if (!text) return;

      cell.textContent = "";
      const mark = document.createElement("span");
      mark.className = `score-mark ${kind}`;
      mark.textContent = text;
      cell.appendChild(mark);
    };

    document.querySelectorAll(".paper-results-table").forEach((table) => {
      const lowerCols = new Set(
        (table.dataset.lowerCols || "")
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean)
          .map((v) => Number(v))
      );

      const rows = [...table.querySelectorAll("tbody tr")];
      if (!rows.length) return;

      const colCount = Math.max(...rows.map((row) => row.cells.length));

      for (let col = 1; col < colCount; col += 1) {
        const entries = rows
          .map((row) => {
            const cell = row.cells[col];
            if (!cell) return null;
            const value = parseValue(cell.textContent);
            if (value === null) return null;
            return { cell, value };
          })
          .filter(Boolean);

        if (!entries.length) continue;

        const ordered = uniqueSorted(entries.map((entry) => entry.value), lowerCols.has(col));
        const best = ordered[0];
        const second = ordered.length > 1 ? ordered[1] : null;

        entries.forEach(({ cell, value }) => {
          if (value === best) {
            cell.classList.add("best-cell");
            applyScoreMark(cell, "best");
          } else if (second !== null && value === second) {
            cell.classList.add("second-cell");
            applyScoreMark(cell, "second");
          }
        });
      }
    });

    // Autoplay silent demos only when visible. A visitor's controls take priority.
    const resultVideos = [...document.querySelectorAll(".result-video")];
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    resultVideos.forEach((video) => {
      video.addEventListener("pointerdown", () => { video.dataset.userControlled = "true"; });
      video.addEventListener("keydown", () => { video.dataset.userControlled = "true"; });
    });
    if ("IntersectionObserver" in window) {
      const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(({ target: video, isIntersecting, intersectionRatio }) => {
          if (!isIntersecting || intersectionRatio < 0.35) {
            video.pause();
          } else if (!reducedMotion.matches && !document.hidden && !video.dataset.userControlled) {
            video.play().catch(() => {});
          }
        });
      }, { threshold: 0.35 });
      resultVideos.forEach((video) => videoObserver.observe(video));
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) resultVideos.forEach((video) => video.pause());
    });
    reducedMotion.addEventListener("change", (event) => {
      if (event.matches) {
        resultVideos.forEach((video) => video.pause());
        document.querySelectorAll(".reveal-pending").forEach((element) => {
          element.classList.remove("reveal-pending");
        });
      }
    });

    // Reveal each lower section once; without JS, every section remains visible.
    if ("IntersectionObserver" in window && !reducedMotion.matches) {
      const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(({ target, isIntersecting }) => {
          if (isIntersecting) {
            target.classList.add("is-visible");
            observer.unobserve(target);
          }
        });
      }, { threshold: 0, rootMargin: "0px 0px -32px 0px" });
      document.querySelectorAll("main > .section:not(.demo-section) > .container").forEach((element) => {
        if (element.getBoundingClientRect().top > window.innerHeight) {
          element.classList.add("reveal-pending");
          revealObserver.observe(element);
        }
      });
    }

    const researchDropdowns = [...document.querySelectorAll(".research-dropdown")];

    document.addEventListener("click", (event) => {
      researchDropdowns.forEach((dropdown) => {
        if (!dropdown.contains(event.target) || event.target.closest(".research-menu a")) {
          dropdown.removeAttribute("open");
        }
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        researchDropdowns.forEach((dropdown) => dropdown.removeAttribute("open"));
      }
    });
  })();
