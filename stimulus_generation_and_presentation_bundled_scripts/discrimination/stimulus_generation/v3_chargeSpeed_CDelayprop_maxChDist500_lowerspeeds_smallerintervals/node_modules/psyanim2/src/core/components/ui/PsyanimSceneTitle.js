import PsyanimComponent from '../../PsyanimComponent.js';

import PsyanimApp from '../../PsyanimApp.js';

export default class PsyanimSceneTitle extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        const sceneNameHeader = document.getElementById("sceneName");

        sceneNameHeader.innerHTML = PsyanimApp.Instance.currentSceneKey;
    }
}