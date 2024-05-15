document.addEventListener('DOMContentLoaded', function () {
    let currentTrackId = parseInt(document.getElementById('current-track-id').value);
    const audioElement = document.getElementById('audio-element');
    const progressBar = document.getElementById('progress-bar');
    const currentTimeElement = document.getElementById('current-time');
    const totalDurationElement = document.getElementById('total-duration');
    const playPauseIcon = document.getElementById('play-pause-icon');
    let isPlaying = false;

    function changeTrack(newTrackId) {
        fetch(`/songs/${newTrackId}`)
            .then(response => response.text())
            .then(html => {
                // Create a temporary DOM element to parse the response HTML
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = html;

                // Extract the track details
                const trackTitle = tempDiv.querySelector('.song-title a').textContent;
                const artistName = tempDiv.querySelector('.artist-name').textContent;
                const albumCover = tempDiv.querySelector('.album-cover').src;
                
                // Update the page content with new song data
                document.querySelector('.song-title a').textContent = trackTitle;
                document.querySelector('.artist-name').textContent = artistName;
                document.querySelector('.album-cover').src = albumCover;
                document.querySelector('.song-title a').href = `/songs/${newTrackId}`;

                // Update the audio element source
                audioElement.src = `/audio/${newTrackId}`;
                audioElement.load();

                // Reset the playback controls
                playPauseIcon.className = 'fas fa-play';
                isPlaying = false;
                audioElement.currentTime = 0;
                progressBar.value = 0;
                currentTimeElement.innerText = '0:00';
                totalDurationElement.innerText = '0:00';

                // Update the current track ID
                currentTrackId = newTrackId;

                // Automatically play the new track
                togglePlayPause();
            })
            .catch(error => console.error('Error fetching song:', error));
    }

    function nextSong() {
        const nextTrackId = currentTrackId + 1; // Simple increment, needs logic adjustment based on actual track list
        changeTrack(nextTrackId);
    }

    function previousSong() {
        const prevTrackId = currentTrackId - 1; // Simple decrement, adjust logic as necessary
        if (prevTrackId >= 0) { // Ensure there is a previous track
            changeTrack(prevTrackId);
        }
    }

    window.nextSong = nextSong; // Make the function available globally
    window.previousSong = previousSong; // Make the function available globally

    function togglePlayPause() {
        if (isPlaying) {
            audioElement.pause();
            playPauseIcon.className = 'fas fa-play';
            isPlaying = false;
        } else {
            audioElement.play();
            playPauseIcon.className = 'fas fa-pause';
            isPlaying = true;
        }
    }

    // Initialize the play/pause toggle
    document.querySelector('.playback-controls button:nth-child(2)').addEventListener('click', togglePlayPause);

    // Event listeners for the audio element
    audioElement.addEventListener('timeupdate', function () {
        const currentTime = audioElement.currentTime;
        const duration = audioElement.duration;
        progressBar.value = (currentTime / duration) * 100;
        currentTimeElement.innerText = formatTime(currentTime);
        totalDurationElement.innerText = formatTime(duration);
    });

    progressBar.addEventListener('input', function () {
        const newTime = (progressBar.value / 100) * audioElement.duration;
        audioElement.currentTime = newTime;
    });

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
});
