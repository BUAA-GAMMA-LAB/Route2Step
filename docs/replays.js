document.querySelectorAll('[data-carousel]').forEach((carousel) => {
  const slides = Array.from(carousel.querySelectorAll('.replay-slide'));
  const dotsContainer = carousel.querySelector('.replay-dots');
  let activeIndex = 0;

  const dots = slides.map((_, index) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'replay-dot';
    dot.setAttribute('aria-label', `Show replay ${index + 1}`);
    dot.addEventListener('click', () => showSlide(index));
    dotsContainer.appendChild(dot);
    return dot;
  });

  function showSlide(index) {
    activeIndex = (index + slides.length) % slides.length;
    slides.forEach((slide, slideIndex) => {
      const isActive = slideIndex === activeIndex;
      slide.classList.toggle('is-active', isActive);
      dots[slideIndex].classList.toggle('is-active', isActive);
      dots[slideIndex].setAttribute('aria-current', isActive ? 'true' : 'false');
      if (!isActive) {
        slide.querySelector('video').pause();
      }
    });
  }

  carousel.querySelector('.replay-prev').addEventListener('click', () => showSlide(activeIndex - 1));
  carousel.querySelector('.replay-next').addEventListener('click', () => showSlide(activeIndex + 1));
  showSlide(0);
});
