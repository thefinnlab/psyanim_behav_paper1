import PsyanimComponent from '../../PsyanimComponent.js';

import PsyanimConstants from "../../PsyanimConstants.js";

export default class PsyanimMouseFollowTarget extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this._radius = 4;

        let bodyOptions = {
            label: 'mouseFollowTarget',
            shape: {
                type: 'circle',
                radius: 4
            },
            collisionFilter: {
                category: PsyanimConstants.COLLISION_CATEGORIES.MOUSE_CURSOR,
                mask: PsyanimConstants.COLLISION_CATEGORIES.NONE
            }
        }

        this.entity.setBody({ type: 'circle', radius: 4 }, bodyOptions);

        // give focus to the canvas so the mouse follow target starts working immediately
        this.entity.scene.game.canvas.focus();
    }

    update(t, dt) {

        super.update(t, dt);

        let x = this.entity.scene.input.x;
        let y = this.entity.scene.input.y;

        if (isFinite(x) && isFinite(y))
        {
            this.entity.x = x;
            this.entity.y = y;    
        }
    }
}