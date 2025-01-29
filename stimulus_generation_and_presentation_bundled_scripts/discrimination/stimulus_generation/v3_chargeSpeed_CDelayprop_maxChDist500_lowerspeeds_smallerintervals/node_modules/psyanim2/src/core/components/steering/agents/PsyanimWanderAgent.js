import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimWanderAgent extends PsyanimComponent {

    vehicle;

    wanderBehavior;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        // compute steering
        let steering = this.wanderBehavior.getSteering();

        this.vehicle.steer(steering);
    }
}