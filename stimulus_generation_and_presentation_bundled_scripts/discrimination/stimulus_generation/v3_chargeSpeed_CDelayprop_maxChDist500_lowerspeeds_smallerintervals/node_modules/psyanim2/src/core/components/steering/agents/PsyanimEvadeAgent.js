import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimEvadeAgent extends PsyanimComponent {

    target;
    evadeBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        let steering = this.evadeBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}