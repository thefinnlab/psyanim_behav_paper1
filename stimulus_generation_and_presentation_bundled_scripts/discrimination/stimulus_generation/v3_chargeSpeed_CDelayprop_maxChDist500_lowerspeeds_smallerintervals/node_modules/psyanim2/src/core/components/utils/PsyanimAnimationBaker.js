import PsyanimComponent from '../../PsyanimComponent.js';

import {
    PsyanimAnimationClip
} from 'psyanim-utils';

export default class PsyanimAnimationBaker extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this._clip = new PsyanimAnimationClip();
        this._state = PsyanimAnimationBaker.STATE.IDLE;

        this._animationParameters = [];
    }

    afterCreate() {

        super.afterCreate();

        // gather component references for each parameter
        this._animationParameters.forEach(parameter => {

            let componentReference = this.entity.getComponent(parameter.componentInfo.type);

            // just store the component reference in the parameter object itself
            parameter.componentReference = componentReference;
        })
    }

    get isRunning() {

        return this._state == PsyanimAnimationBaker.STATE.BAKING;
    }

    addAnimationParameter(parameter) {

        this._animationParameters.push(parameter);
    }

    addAnimationParameters(parameters) {

        this._animationParameters.push(...parameters);
    }

    start() {

        this._state = PsyanimAnimationBaker.STATE.BAKING;
    }

    stop() {

        this._state = PsyanimAnimationBaker.STATE.IDLE;
    }

    clear() {

        this._clip.clear();
    }

    update(t, dt) {

        super.update(t, dt);

        switch (this._state)
        {
            case PsyanimAnimationBaker.STATE.IDLE:

                break;

            case PsyanimAnimationBaker.STATE.BAKING:

                let optionalAttributes = {};

                // update animation parameter values
                this._animationParameters.forEach(parameter => {

                    let componentReference = parameter.componentReference;

                    optionalAttributes[parameter.name] = componentReference[parameter.key];
                });

                // save sample to animation clip
                this._clip.addSample(t, this.entity.position, this.entity.rotation, optionalAttributes);

                break;
        }
    }

    get clip() {

        return this._clip;
    }
}

PsyanimAnimationBaker.STATE = {

    IDLE: 0x0001,
    BAKING: 0x0002,
};