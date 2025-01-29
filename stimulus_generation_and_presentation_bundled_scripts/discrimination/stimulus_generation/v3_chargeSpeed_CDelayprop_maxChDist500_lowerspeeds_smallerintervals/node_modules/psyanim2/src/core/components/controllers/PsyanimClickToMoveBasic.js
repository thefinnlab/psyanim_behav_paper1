import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimClickToMoveBasic extends PsyanimComponent { 

    constructor(entity) {

        super(entity);

        this._arriveTarget = this.scene.addEntity('arriveTarget');

        this._arriveAgent = null;

        this.scene.input.on('pointerup', (pointer) => {

            if (pointer.leftButtonReleased())
            {
                this._arriveTarget.position = new Phaser.Math.Vector2(pointer.x, pointer.y);
            }
        });
    }

    set arriveAgent(value) {

        this._arriveAgent = value;
        this._arriveAgent.target = this._arriveTarget;

        this._arriveTarget.position = this._arriveAgent.entity.position;
    }
}