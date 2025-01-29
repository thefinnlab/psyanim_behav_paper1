import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimVehicle from "..//components/steering/PsyanimVehicle.js";
import PsyanimPreciselyTimedArriveBehavior from "../components/steering/PsyanimPreciselyTimedArriveBehavior.js";
import PsyanimAdvancedFleeBehavior from "../components/steering/PsyanimAdvancedFleeBehavior.js";
import PsyanimSeekBehavior from "../components/steering/PsyanimSeekBehavior.js";
import PsyanimWanderBehavior from "../components/steering/PsyanimWanderBehavior.js";
import PsyanimAdvancedPlayfightBehavior from "../components/steering/PsyanimAdvancedPlayfightBehavior.js";
import PsyanimAdvancedPlayfightAgent from "../components/steering/agents/PsyanimAdvancedPlayfightAgent.js";

/**
 *  Prefab for creating `Advanced Playfight Agents`.
 * 
 *  The `Advanced Playfight Agent` is a variant of the `Playfight Agent` which provides a more precisely
 *  timed charge allowing us to keep frequency of collisions roughly constant throughout the playfight.
 * 
 *  An `Advanced Playfight Agent` has the following components:
 * 
 *  `PsyanimVehicle`, `PsyanimPreciselyTimedArriveBehavior`, `PsyanimAdvancedFleeBehavior`, `PsyanimSeekBehavior`, 
 *  `PsyanimWanderBehavior`, `PsyanimAdvancedPlayfightBehavior`, `PsyanimAdvancedPlayfightAgent`
 */
export default class PsyanimAdvancedPlayfightAgentPrefab extends PsyanimEntityPrefab {

    /** Advanced Playfight Algorithm Params */

    /**
     *  The break duration determines how long, in milliseconds, the agent wanders before attacking its target again.
     *  @type {Number}
     */
    breakDuration;

    /**
     *  Collision frequency determines how often, in milliseconds, the agent should attack it's target. `Collision frequency` = `break duration` + `charge duration`
     * 
     *  Note that this should be longer than the break duration so that the `break duration` + `charge duration` is always less than `collision frequency`.
     * 
     *  @type {Number}
     */
    collisionFrequency;

    /**
     *  Maximum speed the agent can reach while charging towards target.
     *  @type {Number}
     */
    maxChargeSpeed;

    /** Wander params */

    /**
     *  Maximum speed at which the agent will wander.
     *  @type {Number}
     */
    maxWanderSpeed;

    /**
     *  Maximum acceleration the agent can attain during wander.
     *  @type {Number}
     */
    maxWanderAcceleration;

    /**
     *  Radius of the wander circle.
     *  @type {Number}
     */
    wanderRadius;

    /**
     *  Distance the wander circle is offset from the agent's position.
     *  @type {Number}
     */
    wanderOffset;

    /**
     *  Maximum number of degrees the wander target can move around the wander circle per frame.
     *  @type {Number}
     */
    maxWanderAngleChangePerFrame;

    /** Flee Params */

    /**
     *  Maximum speed this agent can flee from the target.
     *  @type {Number}
     */
    maxFleeSpeed;

    /**
     *  Maximum acceleration this agent can reach during flee from target.
     *  @type {Number}
     */
    maxFleeAcceleration;

    /**
     *  Distance from target at which the agent will begin fleeing.
     *  @type {Number}
     */
    panicDistance;

    /** Charge Params */

    /**
     *  Maximum acceleration the agent can attain during a charge.
     *  @type {Number}
     */
    maxChargeAcceleration;

    /**
     *  Distance, in px, from target which the agent will come to rest.
     *  @type {Number}
     */
    innerDecelerationRadius;

    /**
     *  Distance, in px, from target which the agent will begin slowing down.
     *  @type {Number}
     */
    outerDecelerationRadius;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.breakDuration = 2000;
        this.collisionFrequency = 2500;

        this.maxChargeAcceleration = 0.4;

        this.innerDecelerationRadius = 12;
        this.outerDecelerationRadius = 30;

        this.maxWanderSpeed = 4;
        this.maxWanderAcceleration = 0.2;

        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 20;

        this.maxFleeSpeed = 4;
        this.maxFleeAcceleration = 0.2;
        this.panicDistance = 100;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);

        let advancedArrive = entity.addComponent(PsyanimPreciselyTimedArriveBehavior);
        advancedArrive.maxAcceleration = this.maxChargeAcceleration;
        advancedArrive.innerDecelerationRadius = this.innerDecelerationRadius;
        advancedArrive.outerDecelerationRadius = this.outerDecelerationRadius;

        let flee = entity.addComponent(PsyanimAdvancedFleeBehavior);
        flee.maxSpeed = this.maxFleeSpeed;
        flee.maxAcceleration = this.maxFleeAcceleration;
        flee.panicDistance = this.panicDistance;

        let seek = entity.addComponent(PsyanimSeekBehavior);
        seek.maxSpeed = this.maxWanderSpeed;
        seek.maxAcceleration = this.maxWanderAcceleration;

        let wander = entity.addComponent(PsyanimWanderBehavior);
        wander.seekBehavior = seek;
        wander.radius = this.wanderRadius;
        wander.offset = this.wanderOffset;
        wander.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        let playfight = entity.addComponent(PsyanimAdvancedPlayfightBehavior);
        playfight.collisionFrequency = this.collisionFrequency;
        playfight.breakDuration = this.breakDuration;
        playfight.fleeBehavior = flee;
        playfight.advancedArriveBehavior = advancedArrive;
        playfight.wanderBehavior = wander;
        
        let playfightAgent = entity.addComponent(PsyanimAdvancedPlayfightAgent);
        playfightAgent.advancedPlayfightBehavior = playfight;
        playfightAgent.vehicle = vehicle;

        return entity;
    }
}