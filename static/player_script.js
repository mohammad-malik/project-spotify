document.addEventListener('DOMContentLoaded', () => {
    let currentTrackId = parseInt(document.getElementById('current-track-id').value); // Assuming track ID is stored in an input field

    function changeTrack(newTrackId) {
        fetch(`/api/song/${newTrackId}`)
            .then(response => response.json())
            .then(data => {
                if (data) {
                    // Update the page content with new song data
                    document.querySelector('.song-title').textContent = data.track_title;
                    document.querySelector('.artist-name').textContent = data.artist_name;
                    document.querySelector('.album-cover').src = data.track_image_file;
                    document.getElementById('play-pause-icon').className = 'fas fa-play';
                    currentTrackId = newTrackId;
                } else {
                    console.error('Song data not found.');
                }
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
});
