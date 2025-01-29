import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimExperimentControls extends PsyanimComponent {

    experiment;
    networkManager;

    constructor(entity) {

        super(entity);

        // create experiment start button
        this._startButton = document.createElement("button");

        this._startButton.innerHTML = "Run Experiment";

        this._startButton.style = "margin: 2px 5px;"

        this._startButton.onclick = () => this.start();

        // optionally clear out video directory
        this._deleteVideosButton = document.createElement("button");

        this._deleteVideosButton.innerHTML = "Delete Videos";

        this._deleteVideosButton.style = "margin: 2px 5px;"

        this._deleteVideosButton.onclick = () => this.deleteVideos();

        this._parentElement = document.getElementById("experiment-controls");

        this._parentElement.appendChild(this._startButton);

        this._parentElement.appendChild(this._deleteVideosButton);
    }

    start() {

        this.experiment.start();
        this.setControlsEnabled(false);
    }

    deleteVideos() {

        let proceed = confirm(
            "This will delete all videos in the video directory. " + 
            "Are you sure you want to perform this action?");

        if (proceed)
        {
            this.networkManager.doPost('/delete-videos', null);
        }
    }

    setControlsEnabled(enabled) {

        this._startButton.disabled = !enabled;
        this._deleteVideosButton.disabled = !enabled;
    }

    destroy() {

        super.destroy();

        const parentElement = document.getElementById("experiment-controls");

        if (this._startButton)
        {
            parentElement.removeChild(this._startButton);
            
            this._startButton = null;
        }

        if (this._deleteVideosButton)
        {
            parentElement.removeChild(this._deleteVideosButton);

            this._deleteVideosButton = null;
        }
    }
}