document.addEventListener("DOMContentLoaded", function() {
    const flipbook = document.getElementById('flipbook');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const backgroundMusic = document.getElementById('backgroundMusic');
    const volumeBtn = document.getElementById('volumeBtn');
    const imagesFolder = 'static/image/';
    let totalImages = 0; // Adjust this value according to the total number of images
    let currentImageIndex = 0;
    let images = [];
    let currentImage, nextImage;

    console.log("Value from Flask:", value);
    totalImages=value;
  
    function toggleBackgroundMusic() {
        if (backgroundMusic.paused) {
            backgroundMusic.play();
            volumeBtn.classList.add('volume-on'); // Add a class to change volume icon appearance
        } else {
            backgroundMusic.pause();
            volumeBtn.classList.remove('volume-on'); // Remove the class to revert volume icon appearance
        }
    }
  
    // Event listener for clicking the volume icon to toggle background music
    volumeBtn.addEventListener('click', function() {
        toggleBackgroundMusic();
    });
  
    fetchImages();
  
    function fetchImages() {
        for (let i = 1; i <= totalImages; i++) {
            images.push(imagesFolder + i + '.png'); // Adjust this line if your image filenames differ
        }
        loadImages();
    }
  
    function loadImages() {
        for (let i = 0; i < totalImages; i++) {
            const image = createImageElement(images[i]);
            flipbook.appendChild(image);
        }
        showImage(currentImageIndex);
    }
  
    function createImageElement(src) {
        const image = new Image();
        image.src = src;
        image.classList.add('page');
        return image;
    }
  
    function showImage(index) {
        const pages = document.querySelectorAll('.page');
        const previousIndex = (index - 1 + totalImages) % totalImages; // Ensure previous index wraps around
        const nextIndex = (index + 1) % totalImages; // Ensure next index wraps around
  
        const pageFlipSound = document.getElementById('pageFlipSound');
        pageFlipSound.currentTime = 0;
        pageFlipSound.play();
  
        pages.forEach((page, pageIndex) => {
            page.classList.remove('active', 'previous', 'next');
            if (pageIndex === index) {
                page.classList.add('active');
            } else if (pageIndex === previousIndex) {
                page.classList.add('previous');
            } else if (pageIndex === nextIndex) {
                page.classList.add('next');
            }
        });
    }
  
    // Adjust previous and next button event listeners
    prevBtn.addEventListener('click', function() {
        currentImageIndex = (currentImageIndex - 1 + totalImages) % totalImages;
        showImage(currentImageIndex);
    });
  
    nextBtn.addEventListener('click', function() {
        currentImageIndex = (currentImageIndex + 1) % totalImages;
        showImage(currentImageIndex);
    });
  });
  




