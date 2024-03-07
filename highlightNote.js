function highlightNote(noteId, verseText, startPos, endPos) {
    const noteElement = document.getElementById(noteId);
    if(noteElement) {
        const highlightedText = `<span class="highlight">${verseText.substring(startPos, endPos)}</span>`;
        const verseTextElement = document.querySelector('.verse-text');
        if (verseTextElement) {
            verseTextElement.innerHTML = verseText.substring(0, startPos) + highlightedText + verseText.substring(endPos);
        }
    }
}