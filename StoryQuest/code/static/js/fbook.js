// References to DOM Elements
document.addEventListener("DOMContentLoaded", function() {
    const flipbook = document.getElementById('flipbook');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const backgroundMusic = document.getElementById('backgroundMusic');
    const volumeBtn = document.getElementById('volumeBtn');
    const imagesFolder = 'static/images/';
    let totalImages = 0;
    let currentImageIndex = 0;
    let images = [];

    // Fetch text data from text file
    fetch('text.txt')
        .then(response => response.text())
        .then(text => {
            const textData = text.split('\n').map(line => line.trim());
            totalImages = textData.length;
            fetchImages(textData);
        })
        .catch(error => {
            console.error('Error fetching text data:', error);
        });

    function toggleBackgroundMusic() {
        if (backgroundMusic.paused) {
            backgroundMusic.play();
            volumeBtn.classList.add('volume-on');
        } else {
            backgroundMusic.pause();
            volumeBtn.classList.remove('volume-on');
        }
    }

    volumeBtn.addEventListener('click', function() {
        toggleBackgroundMusic();
    });

    function fetchImages(textData) {
        for (let i = 1; i <= totalImages; i++) {
            images.push(imagesFolder + i + '.png');
        }
        loadImages(textData);
    }

    function loadImages(textData) {
        for (let i = 0; i < totalImages; i++) {
            const image = createImageElement(images[i]);
            const text = createTextElement(textData[i]);
            const page = document.createElement('div');
            page.classList.add('page');
            const front = document.createElement('div');
            front.classList.add('front');
            front.appendChild(image);
            const back = document.createElement('div');
            back.classList.add('back');
            back.appendChild(text);
            page.appendChild(front);
            page.appendChild(back);
            flipbook.appendChild(page);
        }
        showImage(currentImageIndex);
    }

    function createImageElement(src) {
        const image = new Image();
        image.src = src;
        image.classList.add('front'); // Adding the front class to images
        return image;
    }

    function createTextElement(text) {
        const textElement = document.createElement('div');
        textElement.classList.add('text');
        textElement.textContent = text;
        textElement.classList.add('back'); // Adding the back class to text
        return textElement;
    }

    function showImage(index) {
        const pages = document.querySelectorAll('.page');
        pages.forEach((page, pageIndex) => {
            page.classList.remove('active', 'previous', 'next');
            if (pageIndex === index) {
                page.classList.add('active');
            } else if (pageIndex === (index - 1 + totalImages) % totalImages) {
                page.classList.add('previous');
            } else if (pageIndex === (index + 1) % totalImages) {
                page.classList.add('next');
            }
        });
    }

    prevBtn.addEventListener('click', function() {
        currentImageIndex = (currentImageIndex - 1 + totalImages) % totalImages;
        showImage(currentImageIndex);
    });

    nextBtn.addEventListener('click', function() {
        currentImageIndex = (currentImageIndex + 1) % totalImages;
        showImage(currentImageIndex);
    });
});
