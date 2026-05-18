document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-slider]").forEach((slider) => {
        const track = slider.querySelector("[data-slider-track]");
        const prevButton = slider.querySelector("[data-slider-prev]");
        const nextButton = slider.querySelector("[data-slider-next]");
        const indicators = Array.from(slider.querySelectorAll("[data-slider-indicator]"));

        let index = 0;

        function getCardsPerView() {
            return Number(slider.dataset.sliderPerView || 5);
        }

        function updateSlider() {
            const items = Array.from(track.children);
            const item = items[0];

            if (!item) {
                return;
            }

            const gap = parseFloat(getComputedStyle(track).gap) || 0;
            const itemWidth = item.getBoundingClientRect().width;
            const step = itemWidth + gap;

            const maxIndex = Math.max(0, items.length - getCardsPerView());

            index = Math.max(0, Math.min(index, maxIndex));

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

        nextButton?.addEventListener("click", () => {
            index += 1;
            updateSlider();
        });

        prevButton?.addEventListener("click", () => {
            index -= 1;
            updateSlider();
        });

        indicators.forEach((indicator) => {
            indicator.addEventListener("click", () => {
                index = Number(indicator.dataset.slideIndex);
                updateSlider();
            });
        });

        window.addEventListener("resize", updateSlider);

        updateSlider();
    });
});