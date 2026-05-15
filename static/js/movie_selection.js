document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("movie-search-input");
    const resultsBox = document.getElementById("movie-search-results");

    const modalElement = document.getElementById("movieConfirmModal");
    const modalContent = document.getElementById("movie-confirm-content");
    const confirmButton = document.getElementById("confirm-movie-button");

    if (!input || !resultsBox) {
        return;
    }

    let debounceTimeout = null;
    let selectedMovieId = null;
    let lastResultsHtml = "";

    input.addEventListener("input", () => {
        const query = input.value.trim();

        clearTimeout(debounceTimeout);

        if (query.length < 2) {
            resultsBox.innerHTML = "";
            return;
        }

        debounceTimeout = setTimeout(async () => {
            const response = await fetch(
                `/api/movies/search/?q=${encodeURIComponent(query)}`
            );

            const html = await response.text();

            lastResultsHtml = html;
            resultsBox.innerHTML = html;
        }, 200);
    });

    resultsBox.addEventListener("click", async (event) => {
        const item = event.target.closest(".search-result");

        if (!item || !item.dataset.movieId) {
            return;
        }

        selectedMovieId = item.dataset.movieId;

        const response = await fetch(`/api/movies/${selectedMovieId}/`);
        modalContent.innerHTML = await response.text();

        const modal = new bootstrap.Modal(modalElement);
        modal.show();

        resultsBox.innerHTML = "";
    });

    confirmButton.addEventListener("click", () => {
        if (!selectedMovieId) {
            return;
        }

        document.getElementById("loading-overlay").classList.remove("d-none");

        setTimeout(() => {
            window.location.href = `/recommendations/?movie_id=${selectedMovieId}`;
        }, 500);
    });

    modalElement.addEventListener("hidden.bs.modal", () => {
        if (lastResultsHtml) {
            resultsBox.innerHTML = lastResultsHtml;
        }
    });
});