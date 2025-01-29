import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimVehicle from "../components/steering/PsyanimVehicle.js";
import PsyanimPreciselyTimedArriveBehavior from "../components/steering/PsyanimPreciselyTimedArriveBehavior.js";
import PsyanimPreciselyTimedArriveAgent from "../components/steering/agents/PsyanimPreciselyTimedArriveAgent.js";

export default class PsyanimPreciselyTimedArriveAgentPrefab extends PsyanimEntityPrefab {

    chargeDuration;
    innerDecelerationRadius;
    outerDecelerationRadius;
    maxAcceleration;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.chargeDuration = 0.9;
        this.innerDecelerationRadius = 10;
        this.outerDecelerationRadius = 30;
        this.maxAcceleration = 0.4;
    }

    create(entity) {

        super.create(entity);

        let vehicle1 = entity.addComponent(PsyanimVehicle);

        let advancedArrive = entity.addComponent(PsyanimPreciselyTimedArriveBehavior);

        advancedArrive.chargeDuration = this.chargeDuration;
        advancedArrive.innerDecelerationRadius = this.innerDecelerationRadius;
        advancedArrive.outerDecelerationRadius = this.outerDecelerationRadius;
        advancedArrive.maxAcceleration = this.maxAcceleration;

        this.advancedArriveAgent = entity.addComponent(PsyanimPreciselyTimedArriveAgent);

        this.advancedArriveAgent.vehicle = vehicle1;
        this.advancedArriveAgent.advancedArriveBehavior = advancedArrive;

        return entity;
    }
}