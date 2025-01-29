import PsyanimComponent from '../../PsyanimComponent.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

export default class PsyanimMimic extends PsyanimComponent {

    /**
     *  Target entity to mimic.
     *  @type {PsyanimEntity}
     */
    target;

    /**
     *  Angle, in degrees, by which to rotate all of the target's displacements before applying them
     *  to this entity.
     *  @type {Number}
     */
    angleOffset;

    constructor(entity) {

        super(entity);

        this.angleOffset = 0;

        this._targetCurrentPosition = null;
        this._targetPreviousPosition = null;
    }

    afterCreate() {

        super.afterCreate();

        if (this.target == null)
        {
            PsyanimDebug.error("PsyanimMimic 'target' is null!");
        }

        this._targetCurrentPosition = this.target.position;
        this._targetPreviousPosition = this.target.position;
    }

    _computeDisplacement() {

        let displacement = this._targetCurrentPosition.clone()
            .subtract(this._targetPreviousPosition);

        displacement.rotate(this.angleOffset * Math.PI / 180.0);

        return displacement;
    }

    update(t, dt) {

        super.update(t, dt);

        this._targetPreviousPosition = this._targetCurrentPosition;

        this._targetCurrentPosition = this.target.position;

        let displacement = this._computeDisplacement();

        this.entity.position = this.entity.position
            .add(displacement);

        this.entity.angle = this.target.angle;
    }
}