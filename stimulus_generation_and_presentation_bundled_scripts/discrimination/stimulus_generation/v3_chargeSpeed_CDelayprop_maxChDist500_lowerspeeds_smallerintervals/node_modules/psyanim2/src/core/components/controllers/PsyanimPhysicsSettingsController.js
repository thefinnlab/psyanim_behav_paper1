import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimPhysicsSettingsController extends PsyanimComponent {

    slowTimeScale;

    constructor(entity) {

        super(entity);

        this.slowTimeScale = 0.05;

        this._keys = {
            T: this.entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.T)
        };
    }

    update(t, dt) {

        super.update(t, dt);

        if (Phaser.Input.Keyboard.JustDown(this._keys.T)) {

            if (this.entity.scene.matter.world.engine.timing.timeScale < 1)
            {
                this.entity.scene.matter.world.engine.timing.timeScale = 1.0;
            }
            else
            {
                this.entity.scene.matter.world.engine.timing.timeScale = this.slowTimeScale;
            }
        }
    }
}