document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("movie-search-input");
    const resultsBox = document.getElementById("movie-search-results");
    const modalElement = document.getElementById("movie-confirm-modal");

    if (!input || !resultsBox) {
        return;
    }

    let modalContent = null;
    if (modalElement) {
        modalContent = modalElement.querySelector(".modal-dialog");
    }

    let debounceTimeout = null;
    let selectedMovieId = null;
    let activeIndex = -1;
    let lastResultsHtml = "";
    let searchController = null;
    let latestSearchQuery = "";
    let searchTimeout = 100;

    function getResults() {
        return Array.from(resultsBox.querySelectorAll(".search-result[data-movie-id]"));
    }

    function setActiveResult(index) {
        const results = getResults();

        results.forEach((item) => {
            item.classList.remove("is-active");
        });

        if (!results.length) {
            activeIndex = -1;
            return;
        }

        activeIndex = Math.max(0, Math.min(index, results.length - 1));

        results[activeIndex].classList.add("is-active");
        results[activeIndex].scrollIntoView({
            block: "nearest",
        });
    }

    async function openMovieModal(item) {
        if (!modalElement || !modalContent) {
            return;
        }

        selectedMovieId = item.dataset.movieId;

        const response = await fetch(`/api/movies/${selectedMovieId}/`);
        modalContent.innerHTML = await response.text();

        resultsBox.innerHTML = "";

        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    }

    input.addEventListener("input", () => {
        const query = input.value.trim();

        clearTimeout(debounceTimeout);
        activeIndex = -1;

        if (searchController) {
            searchController.abort();
        }

        if (query.length < 2) {
            resultsBox.innerHTML = "";
            lastResultsHtml = "";
            return;
        }

        latestSearchQuery = query;

        debounceTimeout = setTimeout(async () => {
            searchController = new AbortController();

            try {
                const response = await fetch(
                    `/api/movies/search/?q=${encodeURIComponent(query)}`,
                    {
                        signal: searchController.signal,
                    }
                );

                const html = await response.text();

                if (query !== latestSearchQuery) {
                    return;
                }

                lastResultsHtml = html;
                resultsBox.innerHTML = html;

                setActiveResult(0);
            } catch (error) {
                if (error.name !== "AbortError") {
                    console.error("Search failed:", error);
                }
            }
        }, searchTimeout);
    });

    input.addEventListener("keydown", async (event) => {
        const results = getResults();

        if (!results.length) {
            return;
        }

        if (event.key === "ArrowDown") {
            event.preventDefault();
            setActiveResult(activeIndex + 1);
        }

        if (event.key === "ArrowUp") {
            event.preventDefault();
            setActiveResult(activeIndex - 1);
        }

        if (event.key === "Enter") {
            event.preventDefault();

            if (activeIndex >= 0) {
                await openMovieModal(results[activeIndex]);
            }
        }
    });

    resultsBox.addEventListener("click", async (event) => {
        const item = event.target.closest(".search-result[data-movie-id]");

        if (!item) {
            return;
        }

        await openMovieModal(item);
    });

    if (modalElement) {
        modalElement.addEventListener("hidden.bs.modal", () => {
            if (lastResultsHtml) {
                resultsBox.innerHTML = lastResultsHtml;
                setActiveResult(0);
            }
        });
    }

    document.addEventListener("click", (event) => {
        const button = event.target.closest("#confirm-movie-button");

        if (!button) {
            return;
        }

        const movieId = button.dataset.movieId || selectedMovieId;

        if (!movieId) {
            return;
        }

        const checkedBoxes = Array.from(document.querySelectorAll('input[name="preferences"]:checked'))
            .map(cb => cb.value);

        let url = new URL(`/recommendations/`, window.location.origin);
        url.searchParams.set("movie_id", movieId);

        if (checkedBoxes.length > 0) {
            url.searchParams.set("prefs", checkedBoxes.join(","));
        }

        window.location.href = url.toString();
    });

    document.addEventListener("click", (event) => {
        const button = event.target.closest("#evaluate-movie-button");

        if (!button) {
            return;
        }

        const movieId = button.dataset.movieId || selectedMovieId;

        if (!movieId) {
            return;
        }

        window.location.href = `/evaluation/?movie_id=${movieId}`;
    });
});