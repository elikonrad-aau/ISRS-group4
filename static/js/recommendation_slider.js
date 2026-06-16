function initSlider(slider) {
    if (slider.dataset.sliderInitialized === "true") {
        return;
    }

    slider.dataset.sliderInitialized = "true";

    const track = slider.querySelector("[data-slider-track]");
    const prevButton = slider.querySelector("[data-slider-prev]");
    const nextButton = slider.querySelector("[data-slider-next]");
    const indicators = Array.from(
        slider.querySelectorAll("[data-slider-indicator]")
    );

    if (!track) {
        return;
    }

    let index = 0;

    function getCardsPerView() {
        return Number(slider.dataset.sliderPerView || 3);
    }

    function updateSlider() {
        const items = Array.from(track.children);
        const item = items[0];

        if (!item) {
            return;
        }

        const gap =
            parseFloat(getComputedStyle(track).gap) || 0;

        const itemWidth =
            item.getBoundingClientRect().width;

        const step = itemWidth + gap;

        const maxIndex = Math.max(
            0,
            items.length - getCardsPerView()
        );

        index = Math.max(
            0,
            Math.min(index, maxIndex)
        );

        track.style.transform = `translateX(${-index * step}px)`;

        if (prevButton) {
            prevButton.disabled = index === 0;
        }

        if (nextButton) {
            nextButton.disabled = index === maxIndex;
        }

        indicators.forEach((indicator, indicatorIndex) => {
            indicator.classList.toggle(
                "is-active",
                indicatorIndex === index
            );
        });
    }

    prevButton?.addEventListener("click", () => {
        index--;
        updateSlider();
    });

    nextButton?.addEventListener("click", () => {
        index++;
        updateSlider();
    });

    indicators.forEach((indicator) => {
        indicator.addEventListener("click", () => {
            index = Number(
                indicator.dataset.slideIndex || 0
            );
            updateSlider();
        });
    });

    window.addEventListener("resize", updateSlider);

    updateSlider();
}

document.addEventListener("DOMContentLoaded", () => {
    document
        .querySelectorAll("[data-slider]")
        .forEach(initSlider);
});

document.body.addEventListener("htmx:load", (event) => {
    const root = event.detail.elt;

    if (root.matches?.("[data-slider]")) {
        initSlider(root);
    }

    root
        .querySelectorAll?.("[data-slider]")
        .forEach(initSlider);
});

function initPopovers(root = document) {
    const popoverTriggerList = root.querySelectorAll(
        '[data-bs-toggle="popover"]'
    );

    popoverTriggerList.forEach((el) => {
        if (el.dataset.popoverInitialized) {
            return;
        }

        el.dataset.popoverInitialized = "true";

        new bootstrap.Popover(el);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initPopovers();
});

document.body.addEventListener("htmx:load", (event) => {
    initPopovers(event.detail.elt);
});