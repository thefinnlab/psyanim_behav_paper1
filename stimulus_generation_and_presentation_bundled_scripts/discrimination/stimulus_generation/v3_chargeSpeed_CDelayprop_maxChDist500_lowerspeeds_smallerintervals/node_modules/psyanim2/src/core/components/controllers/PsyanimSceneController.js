import Phaser from 'phaser';

import PsyanimApp from '../../PsyanimApp.js';
import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimSceneChangeController extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this._keys = {
            J: this.entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.J),
            K: this.entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.K)
        };

        this.events = new Phaser.Events.EventEmitter();
    }

    update(t, dt) {

        super.update(t, dt);

        if (Phaser.Input.Keyboard.JustDown(this._keys.J)) {

            let sceneKeys = PsyanimApp.Instance.sceneKeys;

            let currentSceneKey = PsyanimApp.Instance.currentSceneKey;

            let currentSceneIndex = sceneKeys.indexOf(currentSceneKey);

            let nextSceneIndex = currentSceneIndex - 1;

            if (nextSceneIndex < 0)
            {
                nextSceneIndex = sceneKeys.length - 1;
            }

            PsyanimApp.Instance.loadScene(sceneKeys[nextSceneIndex]);
        }
        else if (Phaser.Input.Keyboard.JustDown(this._keys.K)) {

            let sceneKeys = PsyanimApp.Instance.sceneKeys;

            let currentSceneKey = PsyanimApp.Instance.currentSceneKey;

            let currentSceneIndex = sceneKeys.indexOf(currentSceneKey);

            let nextSceneIndex = currentSceneIndex + 1;

            if (nextSceneIndex == sceneKeys.length)
            {
                nextSceneIndex = 0;
            }

            PsyanimApp.Instance.loadScene(sceneKeys[nextSceneIndex]);
        }
    }
}