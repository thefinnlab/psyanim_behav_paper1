import PsyanimComponent from "../../../PsyanimComponent.js";

export default class PsyanimAdvancedArriveAgent extends PsyanimComponent {

    target;
    advancedArriveBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        if (this.target)
        {
            let steering = this.advancedArriveBehavior.getSteering(this.target);

            this.vehicle.steer(steering);
        }
    }
}