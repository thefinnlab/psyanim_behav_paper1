import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimVehicle from "../components/steering/PsyanimVehicle.js";
import PsyanimSeekBehavior from "../components/steering/PsyanimSeekBehavior.js";
import PsyanimSeekAgent from "../components/steering/agents/PsyanimSeekAgent.js";

export default class PsyanimSeekAgentPrefab extends PsyanimEntityPrefab {

    maxSpeed;
    maxAcceleration;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.maxSpeed = 5;
        this.maxAcceleration = 0.2;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);

        let seek = entity.addComponent(PsyanimSeekBehavior);
        seek.maxSpeed = this.maxSpeed;
        seek.maxAcceleration = this.maxAcceleration;

        let seekAgent = entity.addComponent(PsyanimSeekAgent);

        seekAgent.seekBehavior = seek;
        seekAgent.vehicle = vehicle;

        return entity;
    }
}