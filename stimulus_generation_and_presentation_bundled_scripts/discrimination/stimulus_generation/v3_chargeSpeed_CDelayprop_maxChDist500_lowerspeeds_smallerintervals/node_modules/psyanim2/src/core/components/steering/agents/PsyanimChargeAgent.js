import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimChargeAgent extends PsyanimComponent {

    chargeBehavior;
    vehicle;

    constructor(entity) {

        super(entity);

        this._target = null;
    }

    setTarget(target) {

        this._target = target;
        this.chargeBehavior.computeChargeParams(target);
    }

    update(t, dt) {

        super.update(t, dt);

        let steering = this.chargeBehavior.getSteering(this._target);

        // apply steering manually via euler integration
        let dv = steering.clone().scale(dt);

        let newVelocity = this.entity.velocity.scale(1/16.666)
            .add(dv)
            .scale(16.666);

        this.entity.setVelocity(newVelocity.x, newVelocity.y);

        // adjust orientation
        this.vehicle.lookWhereYoureGoing();
    }
}